import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import copy
import tqdm
import time
import copy
import models.resnet as resnet
import torchvision.transforms.functional as TF
import argparse
import models.vgg as vgg
import models.mobilenet as mobilenet
import models.mobilenetv3 as mobilenetv3
import models.shufflenet as shufflenet

parser = argparse.ArgumentParser()
parser.add_argument("--trans", type=str, default='')
parser.add_argument("--target", type=int, default=None)
parser.add_argument("--patchsize", type=int, default=30)
parser.add_argument("--patchcenter", type=int, default=26)
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--model_name", type=str, default='resnet50')
parser.add_argument("--prior", type=str, default='vgg19')
parser.add_argument("--lamb", type=str, default='1')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

TARGET_INDEX = args.target
PATCHSIZE = args.patchsize
PATCHCENTER = args.patchcenter
IMAGE_SIZE = 32

model_name = args.model_name

def make_log_dir():
    logs = {
        'defense_large_patch_multipatch': args.model_name,
        'patchsize': str(PATCHSIZE) + '-' + str(PATCHCENTER),
        'dataset': args.dataset,
        'epoch': 30,
        'lr': 0.01,
        'trans': args.trans,
        'loss': 'new_gram_patch',
        'prior': args.prior + '_C_mask',
        'lamb': args.lamb,
        'target_index': TARGET_INDEX,
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


def save_image(image_tensor, save_file):
    image_tensor = image_tensor.clone()
    image_tensor = image_tensor[:16]
    image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * 0.5 + 0.5 
    image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * 0.5 + 0.5 
    image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * 0.5 + 0.5 
    torchvision.utils.save_image(image_tensor, save_file, nrow=4)

def pad_transform(patch, image_size, patch_size, offset):
    offset_x, offset_y = offset

    pad = torch.nn.ConstantPad2d((offset_x - patch_size // 2, image_size- patch_size - offset_x + patch_size // 2, offset_y - patch_size // 2, image_size-patch_size-offset_y + patch_size // 2), 0) #left, right, top ,bottom
    
    patchsize_center = PATCHCENTER
    pad_center = torch.nn.ConstantPad2d((offset_x - patchsize_center // 2, image_size- patchsize_center - offset_x + patchsize_center // 2, offset_y - patchsize_center // 2, image_size-patchsize_center-offset_y + patchsize_center // 2), 0)
    mask = torch.ones((3, patch_size, patch_size)).to(device)
    mask_center =  torch.ones((3, patchsize_center, patchsize_center)).to(device)
    return pad(patch), pad(mask) - pad_center(mask_center)


features = []
def viz_collection(module, input):
    x = input[0]
    features.append(x)
        
    
_d = (-30, 30)  if 'd' in args.trans else (0, 0)
_t = (0.05, 0.05)  if 't' in args.trans else (0.0, 0.0)
_c = (0.8, 1.2)  if 'c' in args.trans else (1.0, 1.0)
_r = (-20, 20)  if 'r' in args.trans else (0, 0)

print('AFFINE:', _d, _t, _c, _r)
def affine(img, mask):
    degree, (shift_x, shift_y), scale, (shear_x, shear_y) = transforms.RandomAffine.get_params(_d, _t, _c, _r, (512, 512))
    
    img = TF.affine(img, angle=degree, translate=(shift_x, shift_y), scale=scale, shear=(shear_x, shear_y))
    mask = TF.affine(mask, angle=degree, translate=(shift_x, shift_y), scale=scale, shear=(shear_x, shear_y))
    
    return img, mask
    
def perspective(img, mask):
    startpoint, endpoint = transforms.RandomPerspective.get_params(512, 512, 0.5)
    
    img = TF.perspective(img, startpoints=startpoint, endpoints=endpoint)
    mask = TF.perspective(mask, startpoints=startpoint, endpoints=endpoint)
    
    return img, mask

    
num_classes=43

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
        model.load_state_dict(torch.load('finetunes/vgg19_best.pth'))

    elif model_name == "mobilenet":
        model = mobilenet.MobileNetV2(10).cuda()
        model.load_state_dict(torch.load('finetunes/mobilenet_best_new.pth'))

    elif model_name == "mobilenetv3":
        model = mobilenetv3.MobileNetV3(n_class=10, input_size=32).cuda()
        model.load_state_dict(torch.load('finetunes/mobilenetv3_best_new.pth'))

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

    for param in model.parameters():
        param.requires_grad = False

    return model

def get_inputs(inputs, labels, defense_patch):
    patch, mask = pad_transform(defense_patch, IMAGE_SIZE, PATCHSIZE, (IMAGE_SIZE//2, IMAGE_SIZE//2))
    _patch = patch.index_select(0, labels)
    defense_inputs = inputs * (1 - mask) + _patch * mask
    defense_inputs.clamp_(-1, 1)
    return defense_inputs, _patch * mask


def gen_mask(image_size, patch_size, offset, center):
    print(offset)
    offset_x, offset_y = offset

    pad = torch.nn.ConstantPad2d((offset_x - patch_size // 2, image_size- patch_size - offset_x + patch_size // 2, offset_y - patch_size // 2, image_size-patch_size-offset_y + patch_size // 2), 0) #left, right, top ,bottom
    
    patchsize_center = center
    pad_center = torch.nn.ConstantPad2d((offset_x - patchsize_center // 2, image_size- patchsize_center - offset_x + patchsize_center // 2, offset_y - patchsize_center // 2, image_size-patchsize_center-offset_y + patchsize_center // 2), 0)
    mask = torch.ones((1, patch_size, patch_size)).cuda()
    mask_center =  torch.ones((1, patchsize_center, patchsize_center)).cuda()
    return pad(mask) - pad_center(mask_center)


prior_mask = gen_mask(IMAGE_SIZE, PATCHSIZE, (IMAGE_SIZE//2, IMAGE_SIZE//2), PATCHCENTER)
print(prior_mask.shape)
prior_mask = prior_mask.view(prior_mask.shape[0], -1)

indexs = []
for idx in range(prior_mask.shape[1]):
    if prior_mask[0, idx] > 0:
        indexs.append(idx)
indexs = torch.tensor(indexs).to(device)
def gram(fs):
    _fs = fs.view(fs.shape[0], fs.shape[1], -1)
    _fs = torch.matmul( _fs.transpose(1, 2), _fs)
    # _fs = torch.matmul(_fs, _fs.transpose(1, 2))
    return _fs

def new_gram(fs):
    _image = fs.view(fs.shape[0], fs.shape[1], -1)
    _patch = _image.index_select(2, indexs)

    _fs = torch.matmul(_patch.transpose(1, 2), _image)
    return _fs

    return X_PCA

lamb = [float(x) for x in args.lamb.split('-')]
print(lamb)
def loss_feature(fs, labels):
    new_loss = 0
    _L = len(fs) // 2
    for i in range(_L):
        f1 = new_gram(fs[i][1])
        f2 = new_gram(fs[i + _L][1])
        # print(f1.shape)
        if len(f1.shape) == 4:
            total = f1.shape[0] * f1.shape[1] * f1.shape[2] * f1.shape[3]
        elif len(f1.shape) == 3:
            total = f1.shape[0] * f1.shape[1] * f1.shape[2]

        
        _loss = torch.pow(f1 - f2, 2).sum()  / total
        new_loss += _loss * lamb[i]

    return new_loss / _L



BATCHSIZE = 16

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

print(len(trainset))
print(len(testset))

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

def train_patch(defense_patch, model):
    global features

    defense_patch = torch.autograd.Variable(defense_patch).to(device)
    defense_patch.requires_grad_(True)

    optimizer = torch.optim.Adam([defense_patch], lr=0.03)#, weight_decay=1e-4)
    cirterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        
        total = 0
        correct = 0
        total_loss = 0
        tqdm_bar = tqdm.tqdm(train_loader)
        for i, data in enumerate(tqdm_bar):
            
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batchsize = labels.size(0)
            
            defense_inputs = get_inputs(inputs, labels, defense_patch)[0]
            
            
            features = []
            outputs = model(defense_inputs)
            
            prob = F.softmax(outputs, dim=1)
            prob_index = prob.index_select(1, labels)

            
            loss_f = 0 
            loss_p = cirterion(outputs, labels)
            loss = loss_f + loss_p
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicted = torch.max(outputs, 1)[1]
            correct += (predicted == labels).sum()
            total += labels.size(0)
            total_loss += loss
            
            
            if i % 100 == 0:
                output1 = defense_inputs
                save_image(output1, 'inputs.png')
                output2 = defense_inputs
                save_image(output2, os.path.join(log_dir, 'inputs.png'))
            
            tqdm_bar.set_description("E: %d, loss: %.4f, loss_f: %.6f, loss_p: %.4f, acc: %.4f" % (epoch, total_loss / total, loss_f, loss_p, correct.float() / total))
        
        
    return defense_patch


model_names = args.model_name.split('+')
print(model_names)
models = []
for model_name in model_names:
    model = initialize_model(model_name)
    model.to(device)
    model.eval()
    models.append(model)



if args.prior != 'None':
    defense_patch = torch.load('test_logs/fusion-%s+epoch-5+shape-30-26+loss-CE_mask+/prior.pkl' % (args.prior))
    defense_patch = defense_patch[:, :, 1:31, 1:31]
    defense_patch = torch.autograd.Variable(defense_patch).to(device)
else:
    defense_patch = torch.randn((10, 3, PATCHSIZE, PATCHSIZE)).to(device)
defense_patch.requires_grad_(True)
print(defense_patch.shape)

optimizer_defense_patch = torch.optim.Adam([defense_patch], lr=0.03)#, weight_decay=1e-4)
scheduler_defense_patch = torch.optim.lr_scheduler.MultiStepLR(optimizer_defense_patch, milestones=[3, 8, 15, 25, 40, 60, 90, 140], gamma=1/3)

best_epoch = 0
best_acc = 0
for epoch in range(30):

    defense_patch_s = []
    for model in models:
        defense_patch_ = train_patch(defense_patch=defense_patch.clone(), model=model)
        defense_patch_s.append(defense_patch_)

    total = 0
    correct = 0
    total_loss = 0
    tqdm_bar = tqdm.tqdm(train_loader)
    for i, data in enumerate(tqdm_bar):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batchsize = labels.size(0)
        
        defense_inputs, defense_new_patch = get_inputs(inputs, labels, defense_patch)
        defense_inputs_s = []
        defense_new_patch_s = []
        model_outputs_s = []
        loss_p = 0
        for idx, model in enumerate(models):
            defense_patch_ = defense_patch_s[idx]
            defense_inputs_, defense_new_patch_ = get_inputs(inputs, labels, defense_patch_)
            defense_inputs_s.append(defense_inputs_)
            defense_new_patch_s.append(defense_new_patch_)
            
            
            features = []
            model_outputs = model(defense_inputs)

            prob = F.softmax(model_outputs, dim=1)
            prob_index = prob.index_select(1, labels)
            model_outputs_s.append(model_outputs)


        features = []
        for idx in range(len(defense_inputs_s)):
            features.append((defense_new_patch, defense_inputs))
        for idx in range(len(defense_inputs_s)):
            features.append((defense_new_patch_s[idx], defense_inputs_s[idx]))

        loss_f = loss_feature(features, labels)

        loss = loss_f
        optimizer_defense_patch.zero_grad()
        loss.backward()
        optimizer_defense_patch.step()

        for model_outputs in model_outputs_s:
            predicted = torch.max(model_outputs, 1)[1]
            correct += (predicted == labels).sum().float() / len(models)

        total += labels.size(0)
        total_loss += loss
        
        if i % 100 == 0:
            output1 = defense_inputs
            save_image(output1, 'inputs.png')
            output2 = defense_inputs
            save_image(output2, os.path.join(log_dir, 'inputs.png'))
        
        tqdm_bar.set_description("E: %d, loss: %.4f, loss_f: %.6f, acc: %.4f" % (epoch, total_loss / total, loss_f, correct.float() / total))
    

    with open(os.path.join(log_dir, 'train_log.txt'), 'a') as f:
        f.write("E: %d, loss: %.6f, acc: %.4f\n" % (epoch, total_loss / total, correct.float() / total))
    acc = correct.float() / total
    
    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
        _defense_patch = defense_patch.clamp(-1, 1)
        torch.save(_defense_patch, os.path.join(log_dir, 'defense_patch_best.pkl'))
    
    if epoch % 10 == 9:
        _defense_patch = defense_patch.clamp(-1, 1)
        torch.save(_defense_patch, os.path.join(log_dir, 'defense_patch_%d.pkl'%epoch))
    
    with open(os.path.join(log_dir, 'val_log.txt'), 'a') as f:
        f.write("E: %d, loss: %.6f, acc: %.4f, best_epoch: %d\n" % (epoch, total_loss / total, correct.float() / total, best_epoch))
        
    scheduler_defense_patch.step()