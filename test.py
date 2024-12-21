# coding:utf-8
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models.segmentation as segmentation

from util.MF_dataset import MF_dataset, MF_dataset_extd
from util.util import calculate_accuracy, calculate_result, DEVICE, visual_and_plot, channel_filename

from model import MFNet, SegNet
from train import n_class, data_dir, model_dir, get_model, delete_g
from attack import get_attack
from tqdm import tqdm


def transform_channel(x:Tensor) -> Tensor:
    r, g, b, ir = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]
    r_avg, r_std = r.mean(dim=(1, 2), keepdim=True), r.std(dim=(1, 2), keepdim=True)
    g_avg, g_std = g.mean(dim=(1, 2), keepdim=True), g.std(dim=(1, 2), keepdim=True)
    g_hat = (r - r_avg) / r_std * g_std + g_avg
    return torch.stack([r, g_hat, b, ir], dim=1)
    
    
def main():
    img_dir = os.path.join('data', args.dataset)
    
    if args.model_name == 'SegNet':
        model = eval(args.model_name)(n_class=n_class, in_channels=args.channels)
    elif args.model_name == 'MFNet':
        model = eval(args.model_name)(n_class=n_class)
    else:
        model = segmentation.deeplabv3_resnet50(num_classes=n_class)
        model = get_model(model, args.channels)
        
        model.org_forward = model.forward
        model.forward = lambda x: model.org_forward(x)['out']
   
    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_file, end='')
    model.load_state_dict(torch.load(final_model_file, map_location=DEVICE))
    print('done!')

    assert args.split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
    test_dataset  = MF_dataset_extd(data_dir, args.split, have_label=True, img_dir=img_dir)
    
    if args.single != 0:
        images, labels, names = test_dataset.get_train_item(args.single)
        images = Variable(images).unsqueeze(0)
        labels = Variable(labels).unsqueeze(0)
        if args.model_name == 'DeepLabV3':
            if args.channels == 3:
                images = images[:, :3]
            elif args.channels == 1:
                images = images[:, 3:]

        if args.gpu >= 0:
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

        attack_single = get_attack(args, model)
        images_atk = attack_single(images, labels)
        
        logits = model(images)
        logits_atk = model(images_atk)
        
        pred = logits.argmax(1)
        pred_atk = logits_atk.argmax(1)
        
        images = images.to('cpu').squeeze(0).permute(1, 2, 0).detach().numpy()
        gray = images[..., -1]
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(gray, cmap='inferno')
        ax[0].set_title('Original Image')
    
        plt.show()
    
        visual_and_plot(images, pred, pred_atk)
        
        return
    
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader.n_iter = len(test_loader)

    loss_avg = 0.
    acc_avg  = 0.
    cf = np.zeros((n_class, n_class))

    if args.atk:
        attack = get_attack(args, model)

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.without_g:
                assert args.channels == 3
                images = delete_g(images)
            
            if args.model_name == 'DeepLabV3' and images.shape[1] == 4:
                if args.channels == 3:
                    images = images[:, :3]
                elif args.channels == 1:
                    images = images[:, 3:]
                
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            if args.shuffle:
                images = transform_channel(images)
                
            if args.atk:
                images = attack(images, labels)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            if it % 10 == 0:
                print('|- test iter %s/%s. loss: %.4f, acc: %.4f' \
                        % (it+1, test_loader.n_iter, float(loss), float(acc)))

            predictions = logits.argmax(1)

            for gtcid in range(n_class): 
                for pcid in range(n_class):
                    gt_mask      = labels == gtcid 
                    pred_mask    = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    overall_acc, acc, IoU = calculate_result(cf)
    if args.atk:
        print('| eps: %.4f, alpha: %.4f, steps: %d' % (args.eps, args.alpha, args.steps))
        
    print('| overall accuracy:', overall_acc)
    print('| accuracy of each class:', acc)
    print('| class accuracy avg:', acc.mean())
    print('| IoU:', IoU)
    print('| class IoU avg:', IoU.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='SegNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    parser.add_argument('--single',      '-s',  type=int, default=0)
    parser.add_argument('--channels',    '-c',  type=int, default=4)
    parser.add_argument('--split',       '-sp', type=str, default='test')
    parser.add_argument('--shuffle',            action='store_true')
    parser.add_argument('--dataset',     '-D',  type=str, default='MF', choices=['MF', 'MMIF', 'DIF'])
    parser.add_argument('--without_g',          action='store_true')
    # adv attack
    parser.add_argument('-atk',           action='store_true')
    parser.add_argument('--method',       type=str,   default='MIFGSM', 
                        choices=['PGD', 'FGSM', 'AUTOATTACK', 
                                 'APGD', 'R_FGSM', 'R_PGD',
                                 'EOTPGD', 'MIFGSM', 'CW', 'DeepFool', 'PGDL2',
                                 'pixle', 'FAB'])
    parser.add_argument('--eps',          type=float, default=8/255)
    parser.add_argument('--alpha',        type=float, default=1/255)
    parser.add_argument('--steps',        type=int,   default=16)
    parser.add_argument('--mask_channel', type=int,   default=[], nargs='+', help='channels to mask, list of int')
    parser.add_argument('--norm',         type=str,   default='Linf', choices=['Linf', 'L2'])  
    # adv train
    parser.add_argument('--adv_train',    action='store_true')
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    tmp_model, tmp_optim, final_model, log_name = channel_filename(args.channels, adv_train=args.adv_train, 
                                                                   set_name=args.dataset, no_g=args.without_g)
    final_model_file = os.path.join(model_dir, final_model)
    if args.model_name != 'DeepLabV3':
        assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()
