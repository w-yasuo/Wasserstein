import torch
import argparse
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from loss import L2loss
from model import SqueezeNet
from utile import adjust_learning_rate, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

args = parser.parse_args()

# 加载模型
model = SqueezeNet()

# 定义损失函数以及优化器
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss_func = torch.nn.MSELoss()

# Data loader
traindir = None
valdir = None

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_transforms = transforms.Compose([
    transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize, ])
print('Using image size', image_size)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, val_transforms),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
    image = Image.open(r"D:\TEST_IMAGE\2.jpg")
    image = image.resize((224, 224))
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    image_tensor = image_tensor.reshape(1, 3, 224, 224)
    # print("model:", model)
    output1, output2, output3 = model(image_tensor)
    print("output1:", output1)
    print("output2:", output2)
    print("output3:", output3)
