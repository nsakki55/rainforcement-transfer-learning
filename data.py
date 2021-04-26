from logging import getLogger

import torch
from torchvision import datasets, transforms

logger = getLogger('__main__').getChild('data')

class Image:
    def __init__(self, args):
        # ImageNetの正規化に合わせる
        # https://pytorch.org/docs/master/torchvision/models.html
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if args.pre_trained_network == 'inceptionv3':
            self.size = 299
        else:
            self.size = 224
        
        # PyTorch 公式実装から https://pytorch.org/docs/master/torchvision/models.html
        if args.data_augmentation:
            logger.debug('train with data augmentation')
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])}
        else:
            logger.debug('train without data augmentation')
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])}

        if args.dataset == 'cifar10':
            Dataset = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            Dataset = datasets.CIFAR100
        elif args.dataset == 'mnist':
            Dataset = datasets.MNIST
        else:
            raise NotImplementedError('Unknown datasets:{}'.format(args.dataset))
        
        train_valid_dataset = Dataset(root='data/', train=True, transform=data_transforms['train'], download=True)
        test_dataset = Dataset(root='data/', train=False, transform=data_transforms['val'], download=True)
       
        # Subsetではtrain, validation データはシャッフルされない。https://qiita.com/takurooo/items/ba8c509eaab080e2752c
        n_samples = len(train_valid_dataset)
        train_size = int(n_samples * (1 - args.validation_size))
        
        train_indices = list(range(0,train_size))
        valid_indices = list(range(train_size, n_samples))

        train_dataset = torch.utils.data.dataset.Subset(train_valid_dataset, train_indices)
        valid_dataset = torch.utils.data.dataset.Subset(train_valid_dataset, valid_indices)

        # pin_memory : https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/3
        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True 
        )
        self.valid = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
