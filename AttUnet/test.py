from utils import *
import torch
from tqdm import tqdm
from metrics import calc_iou, eval_seg
from collections import OrderedDict
import argparse
import losses
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from att_unet import AttU_Net

from datasets import LiverDataset
from torch.utils.data import DataLoader

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型名字
    parser.add_argument('--name', default=None,
                        help='model name: (default: model+timestamp)')
    # batch_size 大小
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model', '-a', metavar='model', default='NestedUNet')
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    # loss 默认使用CrossEntropy 损失函数
    parser.add_argument('--loss', default='CrossEntropy')

    # dataset
    parser.add_argument('--dataset', default='liverdataset', help='dataset name')
    # 线程数 默认为0
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'pa': AverageMeter(),
                  'ca': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="val: ", ncols=100)
        for input, target in pbar:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target.squeeze(1).long())
            iou = calc_iou(output, target, config['num_classes'])[1]
            miou, pa, ca, mca = eval_seg(output, target, config['num_classes'])
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['pa'].update(pa, input.size(0))
            avg_meters['ca'].update(ca, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('pa', avg_meters['pa'].avg),
                        ('ca', avg_meters['ca'].avg)])


def main():
    print("====> testing...")
    config = vars(parse_args())
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    # 损失函数
    if config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()
    cudnn.benchmark = True
    # create model
    print("=> creating model %s" % config['model'])
    model = AttU_Net(config['input_channels'], config['num_classes'])
    model = model.to(device)

    model_path = "./models/liverdataset_Unet_TRAIN/model_87.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    val_root = "./data/val"
    val_label = "./data/val_label"
    val_dataset = LiverDataset(val_root, val_label, train=False, transform=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['num_workers'])

    val_log = validate(config, val_dataloader, model, criterion)

    print('loss %.4f - iou %.4f - pixel_accuracy %.4f - class_accuracy %.4f'
          % (val_log['loss'], val_log['iou'], val_log['pa'], val_log['ca']))


if __name__ == '__main__':
    main()
