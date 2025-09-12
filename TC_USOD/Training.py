import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from dataset import get_loader
import math
from Models.USOD_Net import ImageDepthNet
import os
import pytorch_iou
import pytorch_ssim


criterion = nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


### bce_ssim_loss
def bce_ssim_loss(pred, target):
    bce_out = criterion(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    loss = bce_out + ssim_out
    return loss


### bce_iou_loss
def bce_iou_loss(pred, target):
    bce_out = criterion(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

### dice_loss
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def main(local_rank, num_gpus, args):
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    net = ImageDepthNet(args)
    net.train()
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    #loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch
            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                                     Variable(depths.cuda(local_rank, non_blocking=True)), \
                                                     Variable(label_224.cuda(local_rank, non_blocking=True)), \
                                                     Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                              Variable(contour_28.cuda()), \
                                                              Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs_saliency = net(images, depths)

            d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency

            bce_loss1 = criterion(d1, label_224)
            bce_loss2 = criterion(d2, label_112)
            bce_loss3 = criterion(d3, label_56)
            bce_loss4 = criterion(d4, label_28)
            bce_loss5 = criterion(d5, label_14)
            bce_loss6 = criterion(db, label_14)

            iou_loss1 = bce_iou_loss(d1,  label_224)
            iou_loss2 = bce_iou_loss(ud2, label_224)
            iou_loss3 = bce_iou_loss(ud3, label_224)
            iou_loss4 = bce_iou_loss(ud4, label_224)
            iou_loss5 = bce_iou_loss(ud5, label_224)
            iou_loss6 = bce_iou_loss(udb, label_224)

            c_loss1 = bce_ssim_loss(d1,  contour_224)
            c_loss2 = bce_ssim_loss(ud2, label_224)
            c_loss3 = bce_ssim_loss(ud3, label_224)
            c_loss4 = bce_ssim_loss(ud4, label_224)
            c_loss5 = bce_ssim_loss(ud5, label_224)
            c_loss6 = bce_ssim_loss(udb, label_224)

            d_loss1 = dice_loss(d1,   label_224)
            d_loss2 = dice_loss(ud2,  label_224)
            d_loss3 = dice_loss(ud3,  label_224)
            d_loss4 = dice_loss(ud4,  label_224)
            d_loss5 = dice_loss(ud5,  label_224)
            d_loss6 = dice_loss(udb,  label_224)

            BCE_total_loss = bce_loss1 + bce_loss2 + bce_loss3 + bce_loss4 + bce_loss5 + bce_loss6
            IoU_total_loss = iou_loss1 + iou_loss2 + iou_loss3 + iou_loss4 + iou_loss5 + iou_loss6
            Edge_total_loss = c_loss1 + c_loss2 + c_loss3 + c_loss4 + c_loss5 + c_loss6
            Dice_total_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4 + d_loss5 + d_loss6
            total_loss = Edge_total_loss + BCE_total_loss + IoU_total_loss + Dice_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += bce_loss1.cpu().data.item()

            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- bce loss: {3:.6f} --- e loss: {4:.6f}'.format(
                    (whole_iter_num + 1), (i + 1) * args.batch_size / N_train, total_loss.item(), bce_loss1.item(), c_loss1.item()
                    ))


            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(),
                           args.save_model_dir + 'TC_USOD.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss / iter_num, epoch + 1)

