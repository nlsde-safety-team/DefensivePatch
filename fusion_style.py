import torch
from torchvision.models import inception_v3
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from torchvision import datasets, transforms, models
import torchvision
import cv2
import numpy as np
import os
import tqdm
from PIL import Image
import models.vgg as vgg
import models.resnet as resnet
import models.mobilenet as mobilenet
import models.shufflenet as shufflenet

import json
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')
args = parser.parse_args()

def make_log_dir():
    logs = {
        'fusion': args.model,
        'epoch': 5,
        'shape': 8,
        'loss': 'CE_mask',
    }
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'test_logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    return dir_name

log_dir = make_log_dir()    


def pad_transform(patch, image_size, patch_size, offset):
    offset_x, offset_y = offset

    pad = torch.nn.ConstantPad2d((offset_x - patch_size // 2, image_size- patch_size - offset_x + patch_size // 2, offset_y - patch_size // 2, image_size-patch_size-offset_y + patch_size // 2), 0) #left, right, top ,bottom
    
    patchsize_center = 27
    pad_center = torch.nn.ConstantPad2d((offset_x - patchsize_center // 2, image_size- patchsize_center - offset_x + patchsize_center // 2, offset_y - patchsize_center // 2, image_size-patchsize_center-offset_y + patchsize_center // 2), 0)
    mask = torch.ones((3, patch_size, patch_size)).cuda()
    mask_center =  torch.ones((3, patchsize_center, patchsize_center)).cuda()
    return pad(patch), pad(mask) - pad_center(mask_center)

def save_image(image_tensor, save_file):
    copy_image = image_tensor.clone()
    copy_image = copy_image.detach()
    copy_image = copy_image[:16]
    # print(image_tensor.shape)
    copy_image[:, 0, :, :] = copy_image[:, 0, :, :] * 0.5 + 0.5 
    copy_image[:, 1, :, :] = copy_image[:, 1, :, :] * 0.5 + 0.5 
    copy_image[:, 2, :, :] = copy_image[:, 2, :, :] * 0.5 + 0.5 
    torchvision.utils.save_image(copy_image, save_file, nrow=4)


def initialize_model(model_name):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "vgg16":
        """ VGG16
        """
        model = vgg.vgg16_bn(num_classes=10).cuda()
        model.load_state_dict(torch.load('finetunes/vgg16_best_new.pth'))
    
    elif model_name == "vgg19":
        """ VGG19
        """
        model = vgg.vgg19_bn(num_classes=10).cuda()
        model.load_state_dict(torch.load('finetunes/vgg19_best_new.pth'))

    elif model_name == "mobilenet":
        model = mobilenet.MobileNetV2(10).cuda()
        model.load_state_dict(torch.load('finetunes/mobilenet_best_new.pth'))


    elif model_name == "shufflenet":
        model = shufflenet.ShuffleNet(10).cuda()
        model.load_state_dict(torch.load('finetunes/shufflenet_best_new.pth'))
    
    elif model_name == "resnet50":
        """ Resnet50
        """
        model = resnet.resnet50(num_classes=10).cuda()
        model.load_state_dict(torch.load('finetunes/resnet50_best_new.pth'))
        
    else:
        print("Invalid model name, exiting...")
        exit()

    return model



transform = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),
                          ])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

print(len(trainset))
print(len(testset))

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

prior = torch.randn((10, 3, 32, 32)).cuda()
prior.data /= 100
prior.requires_grad_(True)

prior_mask = torch.zeros((1, 1, 32, 32)).cuda()
prior_mask[:, :, 1:31, 1:31] = 1
prior_mask[:, :, 3:29, 3:29] = 0

torchvision.utils.save_image(prior_mask, 'mask.png', nrow=1)

optimizer = torch.optim.Adam([prior], lr=0.001)


model = initialize_model(args.model)
print(model)

for param in model.parameters():
    param.requires_grad = False

    
model.cuda()
model.eval()

IMAGE_SIZE=32
PATCHSIZE = 30

for epoch in range(10):
    
    correct = 0
    total = 0
    _prob = 0
    
    tqdm_bar = tqdm.tqdm(train_loader)
    for i, data in enumerate(tqdm_bar):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        batchsize = labels.size(0)

        
        prior_inputs = prior.index_select(0, labels) * prior_mask


        model(inputs)
        outputs = model(prior_inputs)

        prob = F.softmax(outputs, dim=1)
        prob_index = prob.index_select(1, labels)
        loss1 = (-torch.log(prob_index + 1e-10) * torch.eye(batchsize).cuda()).sum() / batchsize

        loss = loss1 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.max(outputs, 1)[1]
        correct += (predicted == labels).sum()

        probs = torch.nn.functional.softmax(outputs, dim=1)
        for _ in range(labels.shape[0]):
            _prob += probs[_, labels[_]]


        total += labels.shape[0]
 
        tqdm_bar.set_description('acc %.4f, conf %.4f, loss %.4f' % (correct.float()/total, _prob/total, loss))
        
        if i % 50 == 0:
            save_image(prior, os.path.join(log_dir, 'prior.png'))
    torch.save(prior, os.path.join(log_dir, 'prior.pkl'))
        


    
    

    
