import torch
from torchvision.models import inception_v3
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms, models
import torchvision
import cv2
import numpy as np
import os
import tqdm
from PIL import Image
import vgg
import resnet
import mobilenet
import shufflenet

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
    # print(offset)
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


def gram_matrix(input):
    a, b, c, d = input.shape
    features = input.view(a, b, c * d)
    assert features[0][0][1] == features.transpose(1, 2)[0][1][0]

    
    G = torch.matmul(features, features.transpose(1, 2)) / b
    return G

features = []
def get_viz(module, inputs):
#     print(len(inputs))
#     print(inputs[0].shape)
    features.append(inputs[0])
    

transform = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),
                          ])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

print(len(trainset))
print(len(testset))

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

cirterion1 = torch.nn.CrossEntropyLoss()
cirterion2 = torch.nn.MSELoss()

prior = torch.randn((10, 3, 32, 32)).cuda()
prior.data /= 100
prior.requires_grad_(True)

prior_mask = torch.zeros((1, 1, 32, 32)).cuda()
prior_mask[:, :, 12:20, 12:20] = 1
# prior_mask[:, :, 1:31, 1:31] = 1
# prior_mask[:, :, 3:29, 3:29] = 0

torchvision.utils.save_image(prior_mask, 'mask.png', nrow=1)

optimizer = torch.optim.Adam([prior], lr=0.001)


model = initialize_model(args.model)
print(model)

for param in model.parameters():
    param.requires_grad = False

count = 0
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        print(count, name)
        if count in [9]:
            m.register_forward_pre_hook(get_viz)
        count += 1
    
model.cuda()
model.eval()

IMAGE_SIZE=32
PATCHSIZE = 30

for epoch in range(10):
    
    correct = 0
    total = 0
    prob = 0
    
    tqdm_bar = tqdm.tqdm(train_loader)
    for i, data in enumerate(tqdm_bar):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        
#         prior_inputs, mask = pad_transform(prior, IMAGE_SIZE, PATCHSIZE, (IMAGE_SIZE//2, IMAGE_SIZE//2))

        
        prior_inputs = prior.index_select(0, labels) * prior_mask
        # mask = 
#         prior_inputs = prior_inputs * mask
#         print(viz_inputs.shape)

        features = []

        model(inputs)
        outputs = model(prior_inputs)

        loss1 = cirterion1(outputs, labels)
        loss2 = 0
        _L = len(features) // 2
        for l in range(_L):
            
            gram_1 = gram_matrix(features[l])
            gram_2 = gram_matrix(features[l + _L])
            loss2 += cirterion2(gram_1, gram_2)

        loss = loss1 # + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.max(outputs, 1)[1]
        correct += (predicted == labels).sum()

        probs = torch.nn.functional.softmax(outputs)
        for _ in range(labels.shape[0]):
            prob += probs[_, labels[_]]


        total += labels.shape[0]

        tqdm_bar.set_description('acc %.4f, conf %.4f, loss1 %.4f, loss2 %.4f' % (correct.float()/total, prob/total, loss1, loss2))
        
        if i % 50 == 0:
            save_image(prior, os.path.join(log_dir, 'prior.png'))
    torch.save(prior, os.path.join(log_dir, 'prior.pkl'))
        


    
    

    
