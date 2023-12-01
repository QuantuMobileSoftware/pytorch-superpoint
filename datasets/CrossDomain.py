from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
import cv2
import pandas as pd
import albumentations as A

from utils.photometric import ImgAugTransform, customizedTransform
from utils.homographies import sample_homography_np as sample_homography
from utils.utils import inv_warp_image
from utils.utils import compute_valid_mask
from utils.utils import inv_warp_image, inv_warp_image_batch
from dataset_preparation.utils.io import read_jpeg

from settings import COMPRESSED_CROSS_DOMAIN_DIR
from utils.tools import dict_update
from datasets.data_tools import np_to_tensor
from datasets.data_tools import warpLabels
from utils.var_dim import squeezeToNumpy


def get_aug_common(task="train", h=240, w=320):
    aug_common = A.Compose([
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.Perspective(scale=(0.05, 0.1), pad_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomResizedCrop(h, w, scale=(0.1,1), ratio=(h/w, h/w), always_apply=True),
    ], additional_targets={"image1":"image"})
    return aug_common

def get_aug_sep(task="train"):
    if task == "train":
        aug_px = A.Compose([
            A.ChannelShuffle(p=0.5),
            A.ColorJitter(p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Sharpen(p=0.5),
            A.ToGray(always_apply=True),
            A.ToFloat(always_apply=True),
        ])
    else:
        aug_px = A.Compose([
            A.ToGray(always_apply=True),
            A.ToFloat(always_apply=True)
        ])
    return aug_px


class CrossDomain(data.Dataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        'homography_adaptation': {
            'enable': False
        }
    }

    def __init__(self, export=False, transform=None, task='train', **config):
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.device = "cpu"


        assert task in ["train", "val", "test"]
        base_path = Path(COMPRESSED_CROSS_DOMAIN_DIR)
        split = pd.read_csv(base_path/"split.csv").query("split == @task")
        self.task = task
        self.aug_common = get_aug_common(self.task, *config["preprocessing"]["resize"])
        self.aug_sep = get_aug_sep(self.task)

        self.init_var()

        sequence_set = []
        # labels
        self.labels = False
        if self.config["labels"]:
            self.labels = True
            print("load labels from: ", self.config["labels"]+"/"+task)
            for example in split.itertuples():
                pts_lr = Path(self.config['labels'], task, '{}.npz'.format(example.stack_name))
                sample = {'image': str((base_path/example.lr_file).with_suffix(".jpg")), 'image_cross_domain': str((base_path/example.hr_file).with_suffix(".jpg")), 'name': str(example.stack_name), 'points': str(pts_lr)}
                sequence_set.append(sample)
        else:
            for example in split.itertuples():
                sample = {'image': str((base_path/example.hr_file).with_suffix(".jpg")), 'name': example.stack_name, "image_cross_domain": str((base_path/example.lr_file).with_suffix(".jpg"))}
                sequence_set.append(sample)
        self.samples = sequence_set[1000:1010]

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.enable_photo_train = self.config['augmentation']['photometric']['enable']
        self.enable_homo_train = self.config['augmentation']['homographic']['enable']

        self.enable_homo_val = False
        self.enable_photo_val = False

        self.gaussian_label = False
        if self.config['gaussian_label']['enable']:
            self.gaussian_label = True

    def _read_pair(self, lr_file, hr_file):
        # TODO: if lr is smaller than desired, want to keep the resolution of hr crop
        # loose the res for now 'cause easier
        lr = read_jpeg(lr_file)
        hr = cv2.resize(read_jpeg(hr_file), (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return lr, hr

    def points_to_2D(self, pnts, H, W):
        labels = np.zeros((H, W))
        pnts = pnts.astype(int)
        labels[pnts[:, 1], pnts[:, 0]] = 1
        return labels

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor).to(self.device)

        sample = self.samples[index]
        input  = {}
        input.update(sample)
        # image
        lr_image, hr_image = self._read_pair(sample["image_cross_domain"], sample["image"])
        tr = self.aug_common(image=lr_image, image1=hr_image)  # TODO: add keypoints to here if self.labels
        lr_image, hr_image = tr["image"], tr["image1"]

        lr_image = self.aug_sep(image=lr_image)["image"][...,0]
        hr_image = self.aug_sep(image=hr_image)["image"][...,0]
        assert lr_image.shape == hr_image.shape
        imshape = torch.tensor(lr_image.shape, device=self.device)

        lr_image = torch.tensor(lr_image, dtype=torch.float32).to(self.device).view(-1, *lr_image.shape)
        hr_image = torch.tensor(hr_image, dtype=torch.float32).to(self.device).view(-1, *lr_image.shape)

        valid_mask = compute_valid_mask(imshape, inv_homography=torch.eye(3, device=self.device), device=self.device)
        input.update({'image': hr_image, "image_cross_domain": lr_image})
        input.update({'valid_mask': valid_mask})

        if self.config['homography_adaptation']['enable']:  # need for magicpoint_coco_export.yaml
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([
                sample_homography(np.array([2, 2]), shift=-1, **self.config['homography_adaptation']['homographies']['params'])
                for i in range(homoAdapt_iter)
            ])
            ##### use inverse from the sample homography
            homographies = np.stack([np.linalg.inv(homography) for homography in homographies])
            homographies[0,:,:] = np.identity(3)
            ######

            homographies = torch.tensor(homographies, dtype=torch.float32, device=self.device)
            inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

            # images
            warped_img = inv_warp_image_batch(hr_image.squeeze().repeat(homoAdapt_iter,1,1,1), inv_homographies, mode='bilinear', device=self.device).unsqueeze(0)
            warped_img = warped_img.squeeze()
            # masks
            valid_mask = compute_valid_mask(imshape, inv_homography=inv_homographies, erosion_radius=self.config['augmentation']['homographic']['valid_border_margin'], device=self.device)
            input.update({'image': warped_img, 'valid_mask': valid_mask, 'image_2D': hr_image})
            input.update({'homographies': homographies, 'inv_homographies': inv_homographies})

        

        return input

    def __len__(self):
        return len(self.samples)

    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {'photometric': {}}
        aug_par['photometric']['enable'] = True
        aug_par['photometric']['params'] = self.config['gaussian_label']['params']
        augmentation = ImgAugTransform(**aug_par)
        image = image[:,:,np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
