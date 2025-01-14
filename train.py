# coding:utf-8
import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model, MODELS
from util.MF_dataset import MF_dataset, MF_dataset_extd
from util.util import calculate_accuracy, channel_filename, delete_g
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


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


''' ↓↓↓ copy & modify form https://github.com/lok-18/A2RNet/blob/main/train_robust.py '''

def attack(images, labels, model, steps=3, eps=4/255, alpha=1/255):
    images = images.detach().clone()
    labels = labels.detach().clone()

    noise = torch.rand_like(images) * 0.0003
    adv_img = Variable(torch.clamp(images.data + noise.data, 0, 1), requires_grad=True)

    for _ in range(steps):
        logits = model(adv_img)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            adv_img = adv_img.data + alpha * adv_img.grad.data.sign()
            eta = torch.clamp(adv_img.data - images.data, -eps, eps)
            adv_img = images.data + eta

        adv_img = Variable(torch.clamp(adv_img.data,0,1), requires_grad=True)

    return adv_img.detach()

''' ↑↑↑ copy & modify form https://github.com/lok-18/A2RNet/blob/main/train_robust.py '''


def train(epo, model, train_loader, optimizer, adv_train:bool=False):
    lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()

    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)

        if args.without_g:
            assert args.channels == 3
            images = delete_g(images)

        if args.channels == 3:
            images = images[:, :3]
        elif args.channels == 1:
            images = images[:, 3:]

        adv_loss = 0
        if adv_train:
            model.eval()
            adv_images = attack(images, labels, model, args.steps, args.eps, args.alpha)
            model.train()
            adv_logits = model(adv_images)
            adv_loss = F.cross_entropy(adv_logits, labels)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels) + adv_loss
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t - t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                  % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            t += 5

    info = '| epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(info)
    with open(log_file, 'a') as fh:
        fh.write(info)


@torch.inference_mode
def validation(epo, model, val_loader):
    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()

    model.eval()
    for it, (images, labels, names) in enumerate(val_loader):
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)

        if args.without_g:
            assert args.channels == 3
            images = delete_g(images)

        if args.channels == 3:
            images = images[:, :3]
        elif args.channels == 1:
            images = images[:, 3:]

        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
              % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))

    info = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter)
    print(info)
    with open(log_file, 'a') as fh:
        fh.write(info)


def main():
    model = get_model(args.model_name, n_class, args.channels)
    optim = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location='cpu'))
        optim.load_state_dict(torch.load(checkpoint_optim_file, map_location='cpu'))
        print('done!')
    if args.gpu >= 0: model.cuda(args.gpu)

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

        train(epo, model, train_loader, optim, adv_train=args.adv_train)
        validation(epo, model, val_loader)

        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optim.state_dict(), checkpoint_optim_file)
        print('done!')

    os.rename(checkpoint_model_file, final_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet', choices=MODELS)
    parser.add_argument('--dataset',     '-D',  type=str, default='MF', choices=['MF', 'MMIF', 'DIF'])
    parser.add_argument('--channels',    '-c',  type=int, default=4)
    parser.add_argument('--batch_size',  '-B',  type=int, default=8)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    parser.add_argument('--without_g', action='store_true')
    # adv train
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--steps',     type=int,  default=3)
    parser.add_argument('--eps',       type=eval, default=4/255)
    parser.add_argument('--alpha',     type=eval, default=1/255)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    tmp_model, tmp_optim, final_model, log_name = channel_filename(args.channels, adv_train=args.adv_train, set_name=args.dataset, no_g=args.without_g)
    checkpoint_model_file = os.path.join(model_dir, tmp_model)
    checkpoint_optim_file = os.path.join(model_dir, tmp_optim)
    final_model_file      = os.path.join(model_dir, final_model)
    log_file              = os.path.join(model_dir, log_name)

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
