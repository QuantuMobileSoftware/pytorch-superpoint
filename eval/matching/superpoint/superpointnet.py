import torch
import numpy as np
import cv2
from datetime import datetime

from .unet_parts import *
from .utils import box_nms, getPtsFromHeatmap, flattenDetection

from .losses import extract_patches, soft_argmax_2d, do_log
SUPERPOINT_THRESHOLD = 0.05

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, model_path, patch_size=5, nms_dist=4, conf_thresh=SUPERPOINT_THRESHOLD):
        super(SuperPointNet, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh

        self.load_state_dict(torch.load(model_path,
                                        map_location=torch.device('cpu'))['model_state_dict'])


    def forward(self, x: torch.Tensor):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x:Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        # x = x['image']
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        return semi, desc

    def postprocess(self, semi, desc):
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        output = {'semi': semi, 'desc': desc}

        heatmap = flattenDetection(semi)
        # nms
        heatmap_nms_batch = heatmap_to_nms(heatmap, device=self.device, nms_dist=self.nms_dist,
                                           conf_thresh=self.conf_thresh, tensor=True)
        outs = pred_soft_argmax(heatmap_nms_batch, heatmap, patch_size=self.patch_size, device=self.device)
        residual = outs['pred']
        # extract points
        outs = batch_extract_features(desc, heatmap_nms_batch, residual)
        output.update(outs)

        return output


def pred_soft_argmax(labels_2D, heatmap, patch_size, device):
    """

    return:
        dict {'loss': mean of difference btw pred and res}
    """
    outs = {}
    label_idx = labels_2D[...].nonzero()
    patches = extract_patches(label_idx.to(device), heatmap.to(device), patch_size=patch_size)
    patches_log = do_log(patches)
    dxdy = soft_argmax_2d(patches_log, device, normalized_coordinates=False)
    dxdy = dxdy.squeeze(1)
    dxdy = dxdy - patch_size // 2

    outs['pred'] = dxdy
    outs['patches'] = patches
    return outs


def sample_desc_from_points(coarse_desc, pts, cell_size=8):
    """
    inputs:
        coarse_desc: tensor [1, 256, Hc, Wc]
        pts: tensor [N, 2] (should be the same device as desc)
    return:
        desc: tensor [1, N, D]
    """
    # --- Process descriptor.
    samp_pts = pts.transpose(0, 1)
    H, W = coarse_desc.shape[2] * cell_size, coarse_desc.shape[3] * cell_size
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = torch.ones((1, 1, D))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()

        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.squeeze(2).squeeze(0).transpose(0, 1).unsqueeze(0)
    return desc


def heatmap_to_nms(heatmap, device, nms_dist, conf_thresh, tensor=False, boxnms=False):
    """
    return:
      heatmap_nms_batch: np [batch, 1, H, W]
    """
    to_floatTensor = lambda x: torch.from_numpy(x).type(torch.FloatTensor)
    heatmap_np = toNumpy(heatmap)
    if boxnms:
        heatmap_nms_batch = [box_nms(h.detach().squeeze(), nms_dist, min_prob=conf_thresh)  for h in heatmap]
        heatmap_nms_batch = torch.stack(heatmap_nms_batch, dim=0).unsqueeze(1)
    else:
        heatmap_nms_batch = [heatmap_nms(h, nms_dist, conf_thresh) for h in heatmap_np]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        heatmap_nms_batch = heatmap_nms_batch[:, np.newaxis, ...]
        if tensor:
            heatmap_nms_batch = to_floatTensor(heatmap_nms_batch)
            heatmap_nms_batch = heatmap_nms_batch.to(device)
    return heatmap_nms_batch


def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
    """
    input:
        heatmap: np [(1), H, W]
    """
    heatmap = heatmap.squeeze()
    pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)

    semi_thd_nms_sample = np.zeros_like(heatmap)
    semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1

    return semi_thd_nms_sample


def batch_extract_features(desc, heatmap_nms_batch, residual):
    """
    return: -- type: tensorFloat
      pts: tensor [batch, N, 2] (no grad)  (x, y)
      pts_desc: tensor [batch, N, 256] (grad)
    """
    batch_size = heatmap_nms_batch.shape[0]

    pts_int, pts_desc = [], []
    pts_idx = heatmap_nms_batch[...].nonzero()
    for i in range(batch_size):
        mask_b = (pts_idx[:, 0] == i)
        pts_int_b = pts_idx[mask_b][:, 2:].float()
        pts_int_b = pts_int_b[:, [1, 0]]
        res_b = residual[mask_b]
        pts_b = pts_int_b + res_b
        pts_desc_b = sample_desc_from_points(desc[i].unsqueeze(0), pts_b).squeeze(0)
        pts_int.append(pts_int_b)
        pts_desc.append(pts_desc_b)

    pts_int = torch.stack(pts_int, dim=0)
    pts_desc = torch.stack(pts_desc, dim=0)
    return {'pts_int': pts_int, 'pts_desc': pts_desc}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet().to(device)
    model.eval()
    img = cv2.imread("planet.tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (320, 240))
    image = img.copy()

    img = img.reshape((1, 1, 240, 320))
    img = img / 255
    img = torch.from_numpy(img)
    outs_post = model(img.float().to(device))

    print(outs_post['pts_desc'].shape)
    print(outs_post['pts_int'].shape)
    print(outs_post['pts_int'][0][:, 0].max())
    points = outs_post['pts_int'][0]
    for item in points:
        image = cv2.circle(image, (int(item[0]), int(item[1])), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("/home/quantum/resized_21.jpg", image)
