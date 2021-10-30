import argparse
import os
from collections import OrderedDict
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from metrics import eval_seg

import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from networks.vit_seg_modeling import VisionTransformer

from datasets import LiverDataset
from torch.utils.data import DataLoader

from utils import AverageMeter, str2bool,DiceLoss
from metrics import calc_iou
import losses
from networks.vit_seg_modeling import CONFIGS

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
LOSS_NAMES.append('CrossEntropyLoss')

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置参数
def parse_args():
    parser = argparse.ArgumentParser()
    # 模型名字
    parser.add_argument('--name', default=None,
                        help='model name: (default: model+timestamp)')
    # epoch 选择
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    # batch_size 大小
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model', '-a', metavar='model', default='TransUnet')
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    # loss 默认使用CrossEntropy 损失函数
    parser.add_argument('--loss', default='CrossEntropy', choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='kvasir_data', help='dataset name')

    # 线程数 默认为0
    parser.add_argument('--num_workers', default=0, type=int)

    # optimizer 默认Adam
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    # 最小学习率
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)

    # 提前停止
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


# 训练函数
def train(config, train_loader, model, criterion, criterion_dice,optimizer, epoch=1):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'pc': AverageMeter(),
                  'dice': AverageMeter(),
                  'precision': AverageMeter(),
                  'recall': AverageMeter(),
                  'fpr': AverageMeter()}

    model.train()

    pbar = tqdm(train_loader, desc="train: ", ncols=100)
    for input,target in pbar:
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = 0.5*criterion(output, target.squeeze(1).long())+0.5*criterion_dice(output,target.squeeze(1).long(),softmax=True)
        out = output

        # 前景iou
        iou = calc_iou(out, target, config['num_classes'])[1]
        miou, pc, recall, fpr, precision, mac, dice = eval_seg(out, target, config['num_classes'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 平均损失
        avg_meters['loss'].update(loss.item(), input.size(0))
        # ACC
        avg_meters['pc'].update(pc, input.size(0))
        # 平均IOU
        avg_meters['iou'].update(iou, input.size(0))
        # 平均dice
        avg_meters['dice'].update(dice[1], input.size(0))
        # 平均precision
        avg_meters['precision'].update(precision, input.size(0))
        # 平均recall
        avg_meters['recall'].update(recall, input.size(0))
        # 平均fpr
        avg_meters['fpr'].update(fpr, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('pc', avg_meters['pc'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('fpr', avg_meters['fpr'].avg)
                        ])


# 验证
def validate(config, val_loader, model, criterion,criterion_dice):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="val: ", ncols=100)
        for input, target in pbar:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = 0.5*criterion(output, target.squeeze(1).long())+0.5*criterion_dice(output,target.squeeze(1).long(),softmax=True)
            iou = calc_iou(output, target, config['num_classes'])[1]

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s_TRAIN' % (config['dataset'], config['model'])
    os.makedirs('./models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 将配置参数写入到文件中
    with open('./models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # 损失函数
    if config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
        # 标签平滑策略
        # criterion = losses.CrossEntropyLabelSmooth(config['num_classes'])
    elif config['loss'] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True
    criterion_dice = DiceLoss(n_classes=2)
    # create model
    print("=> creating model %s" % config['model'])

    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    model = VisionTransformer(config_vit, 256, 2)
    model = model.to(device)

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 获取数据
    train_root = "../kvasir/train"
    train_label = "../kvasir/train_label"
    liver_dataset = LiverDataset(train_root, train_label, train=True, transform=True)
    dataloaders = DataLoader(liver_dataset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['num_workers'])

    val_root = "../kvasir/test"
    val_label = "../kvasir/test_label"
    val_dataset = LiverDataset(val_root, val_label, train=False, transform=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['num_workers'])
    # 日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('pc', []),
        ('dice', []),
        ('precision', []),
        ('recall', []),
        ("fpr", []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print('\nEpoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, dataloaders, model, criterion,criterion_dice, optimizer)
        val_log = validate(config, val_dataloader, model, criterion,criterion_dice)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('\nloss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['pc'].append(train_log['pc'])
        log['dice'].append(train_log['dice'])
        log['precision'].append(train_log['precision'])
        log['recall'].append(train_log['recall'])
        log['fpr'].append(train_log['fpr'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        # 将日志写入文件中
        pd.DataFrame(log).to_csv('./models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou and val_log['iou'] > 0.74:
            torch.save(model.state_dict(), './models/%s/model_%d.pth' %
                       (config['name'], epoch))
            best_iou = val_log['iou']
            print("\n=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("\n=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
