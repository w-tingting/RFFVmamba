## -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.distributed as dist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtingting/Downloads/traffic_sign/deep_rff_pytorch/demo')
sys.path.append(ROOT_DIR)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from datasets.gtsrb_dataset import GTSRB
from datasets.btsc_dataset import BelgiumTS
from datasets.ctsd_dataset import CTSD

normalize = transforms.Normalize((0.3403, 0.3121, 0.3214),
                                 (0.2724, 0.2608, 0.2669))
__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class ReshapeTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        return image.view(-1)


class OneHot:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, label):
        y = torch.zeros(self.classes)
        y[label] = 1
        return y

def init_dataset(args):
    if args.dataset == 'GTSRB':
        # Create Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        # Create Datasets
        trainset = GTSRB(root_dir='./datasets/signs', train=True, transform=transform,
                         target_transform=OneHot(classes=args.num_classes))
        testset = GTSRB(root_dir='./datasets/signs', train=False, transform=transform,
                        target_transform=OneHot(classes=args.num_classes))


    elif args.dataset == 'CTSD':
        # Create Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        # Create Datasets
        trainset = CTSD(root_dir='./datasets/signs', train=True, transform=transform,
                         target_transform=OneHot(classes=args.num_classes))
        testset = CTSD(root_dir='./datasets/signs', train=False, transform=transform,
                        target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'BTSC':
        # Create Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        # Create Datasets
        trainset = BelgiumTS(root_dir='./datasets/signs', train=True, transform=transform,
                        target_transform=OneHot(classes=args.num_classes))
        testset = BelgiumTS(root_dir='./datasets/signs', train=False, transform=transform,
                       target_transform=OneHot(classes=args.num_classes))

    else:
        print("key {} not found".format(args.dataset))

    # DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

    # val_size = int(args.validation_split * len(trainset))
    # train_size = len(trainset) - val_size
    # trainset, valset = random_split(trainset, [train_size, val_size])

    # DistributedSampler
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)

    # # BatchSampler
    # train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    # Load Datasets
    # trainloader = DataLoader(trainset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=0)
    # testloader = DataLoader(testset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=True, num_workers=0)
    # valloader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler, pin_memory=True, num_workers=0)

    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    # valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_sampler, trainset, testset

