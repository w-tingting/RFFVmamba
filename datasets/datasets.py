## -*- coding: utf-8 -*-
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = os.path.abspath('/home/wtingting/Downloads/traffic_sign/deep_rff_pytorch/demo')
sys.path.append(ROOT_DIR)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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


# MNIST
def init_dataset(args):
    if args.dataset == 'MNIST':
        # train
        train_dataset = datasets.MNIST(root='data', train=True, download=True, \
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           # transforms.Resize((64, 64)),
                                           # ReshapeTransform()
                                           # transforms.Normalize((0.1307,),(0.3081,))
                                       ]),
                                       target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.MNIST(root='data', train=False, download=True, \
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          # transforms.Resize((64, 64)),
                                          # ReshapeTransform()
                                          # transforms.Normalize((0.1307,),(0.3081,))
                                      ]),
                                      target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'FMNIST':
        # train
        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, \
                                              transform=transforms.Compose([
                                                  transforms.ToTensor()
                                                  # ReshapeTransform()
                                                  # transforms.Normalize((0.1307,),(0.3081,))
                                              ]),
                                              target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, \
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()

                                                 # transforms.Normalize((0.1307,),(0.3081,))
                                             ]),
                                             target_transform=OneHot(classes=args.num_classes))

    elif args.dataset == 'CIFAR10':

        # train
        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, \
                                         transform=transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(.4, .4, .4),
                                             transforms.ToTensor(),
                                             Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                                             normalize,
                                         ]),
                                         target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, \
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                            # ReshapeTransform()
                                            # transforms.Normalize((0.1307,),(0.3081,))
                                        ]),
                                        target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'CIFAR100':
        # train
        train_dataset = datasets.CIFAR100(root='data', train=True, download=True, \
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Resize((56, 56)),
                                              transforms.ColorJitter(.4, .4, .4),
                                              transforms.ToTensor(),
                                              Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                                              normalize,
                                          ]),
                                          target_transform=OneHot(classes=args.num_classes))

        # test
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, \
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Resize((56, 56)),
                                             # ReshapeTransform()
                                             # transforms.Normalize((0.1307,),(0.3081,))
                                         ]),
                                         target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'tiny-imagenet-200':
        train_path = '/home/wtingting/Downloads/traffic_sign/demo/datasets/tiny-imagenet-200/train/images'
        test_path = '/home/wtingting/Downloads/traffic_sign/demo/datasets/tiny-imagenet-200/val/images'

        train_dataset = datasets.ImageFolder(train_path,
                                             transform=transforms.Compose([
                                                 transforms.Resize((56, 56)),
                                                 transforms.ToTensor(),
                                             ]), target_transform=OneHot(classes=args.num_classes))

        # print("*"*100)
        # print(train_dataset.class_to_idx)
        test_dataset = datasets.ImageFolder(test_path,
                                            transform=transforms.Compose([
                                                transforms.Resize((56, 56)),
                                                transforms.ToTensor(),
                                            ]), target_transform=OneHot(classes=args.num_classes))
    elif args.dataset == 'miniimagenet':
        train_path = '/home/wtingting/Downloads/traffic_sign/demo/datasets/miniImagenet/ILSVRC2015/Data/CLS-LOC/train'
        test_path = '/home/wtingting/Downloads/traffic_sign/demo/datasets/miniImagenet/ILSVRC2015/Data/CLS-LOC/val'

        train_dataset = datasets.ImageFolder(train_path, transform=transforms.Compose([
                                                 transforms.Resize((56, 56)),
                                                 transforms.ToTensor(),
                                             ]), target_transform=OneHot(classes=args.num_classes))

        # print("*"*100)
        # print(train_dataset.class_to_idx)
        test_dataset = datasets.ImageFolder(test_path, transform=transforms.Compose([
                                                 transforms.Resize((56, 56)),
                                                 transforms.ToTensor(),
                                             ]), target_transform=OneHot(classes=args.num_classes))

    elif args.dataset == 'EuroSAT':
        # else:
        path = '/home/wtingting/Documents/deep_rff_pytorch/demo/datasets/EuroSAT'
        train_dataset = datasets.ImageFolder(path, transform=transforms.ToTensor(),
                                             target_transform=OneHot(classes=args.num_classes))
        # print(train_dataset.class_to_idx)
        test_dataset = None

    else:
        path = '/home/wtingting/Documents/deep_rff_pytorch/demo/datasets/{}'.format(args.dataset)
        train_dataset = datasets.ImageFolder(path,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(), transforms.Resize((128, 128))]),
                                             target_transform=OneHot(classes=args.num_classes))
        test_dataset = None

    if test_dataset is None:
        train_size = int(0.7 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    val_size = int(args.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader, test_loader, val_loader

# dataset = datasets.ImageFolder('/home/wtt/deep_rff_pytorch/demo/datasets/EuroSAT', transform=transforms.ToTensor(),
#                                target_transform=OneHot(classes=10))
# train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
# for i, (data, target) in enumerate(train_loader):
#     print(data)
#     print(target)
# # print(dataset.classes)
# # print(dataset.class_to_idx)
# # print(dataset.imgs)
# img = dataset[0]
# print(dataset[0][0].shape)
# print(dataset[0])
