# coding:utf-8
import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MFNet, SegNet, get_dv3_model  # keep for eval()
from attack import get_attack
from util.MF_dataset import MF_dataset, MF_dataset_extd
from util.util import DEVICE, calculate_accuracy, channel_filename, delete_g
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise


# config
n_class   = 9
data_dir  = 'data/MF/'
model_dir = 'weights/'
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0), 
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
lr_start  = 0.01
lr_decay  = 0.95

def train(epo, model, train_loader, optimizer, adv_train:bool=False):
    lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()

    if adv_train:
        atk = get_attack(args, model)

    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = images.cuda(args.gpu) 
        labels = labels.cuda(args.gpu)

        if args.without_g:
            assert args.channels == 3
            images = delete_g(images)

        if args.model_name == 'DeepLabV3':
            if args.channels == 3:
                images = images[:, :3]
            elif args.channels == 1:
                images = images[:, 3:]

        if args.gpu >= 0:
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

        if adv_train:
            model.eval()
            images = atk(images, labels)
            model.train()

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            t += 5

    content = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


@torch.inference_mode
def validation(epo, model, val_loader):
    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()
    
    model.eval()
    for it, (images, labels, names) in enumerate(val_loader):
        if args.without_g:
            assert args.channels == 3
            images = delete_g(images)
        
        if args.model_name == 'DeepLabV3':
            if args.channels == 3:
                images = images[:, :3]
            elif args.channels == 1:
                images = images[:, 3:]
            
        if args.gpu >= 0:
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

        logits = model(images)
            
        loss = F.cross_entropy(logits, labels)
        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))

    content = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' \
            % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


def main():
    if args.model_name == 'SegNet':
        model = eval(args.model_name)(n_class=n_class, in_channels=args.channels)
    elif args.model_name == 'MFNet':
        model = eval(args.model_name)(n_class=n_class)
    elif args.model_name == 'DeepLabV3':
        model = get_dv3_model(n_class=n_class, in_channels=args.channels)
    else: raise ValueError(f'unknown model: {args.model_name}')

    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005) 

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location=DEVICE))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')

    img_dir = os.path.join('data', args.dataset)
    train_dataset = MF_dataset_extd(data_dir, 'train', have_label=True, img_dir=img_dir, transform=augmentation_methods)
    val_dataset   = MF_dataset_extd(data_dir, 'val',   have_label=True, img_dir=img_dir)

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)

    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)

        train(epo, model, train_loader, optimizer, adv_train=args.adv_train)
        validation(epo, model, val_loader)

        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

    os.rename(checkpoint_model_file, final_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet', choices=['SegNet', 'MFMet', 'DeepLabV3'])
    parser.add_argument('--batch_size',  '-B',  type=int, default=8)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    parser.add_argument('--channels',    '-c',  type=int, default=4)
    parser.add_argument('--dataset',     '-D',  type=str, default='MF', choices=['MF', 'MMIF', 'DIF'])
    parser.add_argument('--without_g', action='store_true')
    # adv_train
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--method',    type=str,   default='PGD', choices=['PGD'])
    parser.add_argument('--eps',       type=float, default=8/255)
    parser.add_argument('--alpha',     type=float, default=1/255)
    parser.add_argument('--steps',     type=int,   default=10)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    tmp_model, tmp_optim, final_model, log_name = channel_filename(args.channels, adv_train=args.adv_train, 
                                                                   set_name=args.dataset, no_g=args.without_g)
    checkpoint_model_file = os.path.join(model_dir, tmp_model)
    checkpoint_optim_file = os.path.join(model_dir, tmp_optim)
    final_model_file      = os.path.join(model_dir, final_model)
    log_file              = os.path.join(model_dir, log_name)

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
