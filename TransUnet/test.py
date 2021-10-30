from utils import *
import torch
from tqdm import tqdm
from metrics import calc_iou,eval_seg,eval_hausdorff
from collections import OrderedDict
import argparse
import losses
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer,CONFIGS

from metric_aji import calculate_AJI
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
    parser.add_argument('--model', '-a', metavar='model', default='TransUnet')
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')



    # loss 默认使用CrossEntropy 损失函数
    parser.add_argument('--loss', default='CrossEntropy')

    # dataset
    parser.add_argument('--dataset', default='liverdataset',help='dataset name')
    # 线程数 默认为0
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

def validate(config,val_loader,model,criterion):

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'pc': AverageMeter(),
                  'recall': AverageMeter(),
                  'precision':AverageMeter(),
                  'mac':AverageMeter(),
                  'dice':AverageMeter(),
                  'fpr':AverageMeter(),
                  'hd':AverageMeter(),
                  'aji':AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(val_loader,desc="val: ",ncols=100)
        for input,target in pbar:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output,target.squeeze(1).long())
            iou = calc_iou(output,target,config['num_classes'])[1]
            miou,pc,recall,fpr,precision,mac,dice = eval_seg(output, target, config['num_classes'])
            hd = eval_hausdorff(output,target)
            aji = calculate_AJI(output,target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['pc'].update(pc,input.size(0))
            avg_meters['recall'].update(recall,input.size(0))
            avg_meters['precision'].update(precision,input.size(0))
            avg_meters['mac'].update(mac,input.size(0))
            avg_meters['dice'].update(dice[1],input.size(0))
            avg_meters['fpr'].update(fpr,input.size(0))
            avg_meters['hd'].update(hd,input.size(0))
            avg_meters['aji'].update(aji,input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('pc',avg_meters['pc'].avg),
                        ('recall',avg_meters['recall'].avg),
                        ('precision',avg_meters['precision'].avg),
                        ('mac',avg_meters['mac'].avg),
                        ('dice',avg_meters['dice'].avg),
                        ('fpr',avg_meters['fpr'].avg),
                        ('hd',avg_meters['hd'].avg),
                        ('aji',avg_meters['aji'].avg)])


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
    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    model = VisionTransformer(config_vit, 256, 2)
    model = model.to(device)

    # model_path = "./models/liver_data_TransUnet_TRAIN/model_272.pth"
    # model_path = "./models/cvcdb_data_TransUnet_TRAIN/model_293.pth"
    model_path = "./models/kvasir_data_TransUnet_TRAIN/model_232.pth"

    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    test_root = "../kvasir/test"
    test_label = "../kvasir/test_label"
    test_dataset = LiverDataset(test_root,test_label,train=False,transform=False)
    val_dataloader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=config['num_workers'],drop_last=True)

    val_log = validate(config,val_dataloader,model,criterion)

    # print(val_log['dice'],val_log['mca'])
    print('loss %.4f - iou(JA) %.4f - pixel_accuracy(AC) %.4f - recall %.4f - precision %.4f - dice %.4f - sp %.4f - hd %.4f - aji %.4f'
              % (val_log['loss'], val_log['iou'], val_log['pc'], val_log['recall'],val_log['precision'],val_log['dice'],val_log['fpr'],val_log['hd'],val_log['aji']))


if __name__ == '__main__':
    main()
