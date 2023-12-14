import argparse
import yaml
import os
import logging


import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

from utils.utils import getWriterPath

from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *

###### util functions ######
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass

from Train_model_heatmap import Train_model_heatmap


if __name__ == "__main__":
    with open("/home/topkech/work/pytorch-superpoint/configs/cross_domain_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert 'train_iter' in config

    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)

    writer = SummaryWriter(getWriterPath(task="./debug", exper_name="debug", date=True))
    ## save data
    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')
    # init the training agent using config file
    train_agent = Train_model_heatmap(config, save_path="./debug", device=device)

    # writer from tensorboard
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    train_agent.dataParallel()

    for sample in train_loader:
        print(sample.keys())
        # sample["warped_valid_mask"] = torch.ones_like(sample["warped_valid_mask"])
        # sample["valid_mask"] = torch.ones_like(sample["valid_mask"])
        # sample["labels_2D_gaussian"] = torch.zeros_like(sample["labels_2D_gaussian"])
        # sample["warped_labels_gaussian"] = torch.zeros_like(sample["warped_labels_gaussian"])
        train_agent.train_val_sample(sample, train=True)
        exit(0)
