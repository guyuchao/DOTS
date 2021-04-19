import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search_imagenet import Network
import random
import torchvision.transforms as transforms
from engine import Engine
from torch.utils.data import ConcatDataset

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_dir', type=str, default='/home/guyuchao/SubImageNet', help='location of the data corpus')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=70, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--op_search_layers', type=int, default=14, help='total number of layers')
parser.add_argument('--tp_search_layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='imagenet', help='experiment name')
parser.add_argument('--seed', type=int, default=random.randint(1, 10000), help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--Tmax', type=float, default=10, help='learning rate for arch encoding')
parser.add_argument('--Ttpmin', type=float, default=0.02, help='weight decay for anneal topology')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
args = parser.parse_args()

engine = Engine(args)
if engine.local_rank == 0:
    args.save = 'logs/search/search-{}-{}'.format(args.save, args.seed)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

CLASSES = 1000

# define four stages
Op_Pretrain_Start = 0
Op_Search_Start = 20
Tp_Pretrain_Start = 30
Tp_Search_Start = 50

Tp_Anneal_Rate=pow(args.Ttpmin/args.Tmax,1.0/(args.epochs-Tp_Search_Start-1))


def print_genotype(model):

    genotype, weight_normal, weight_reduce = model.module.genotype()

    logging.info('genotype = %s', genotype)
    logging.info("normal:")
    logging.info(weight_normal)
    logging.info("reduce:")
    logging.info(weight_reduce)
    if 'tp' in model.module.phase:
        normal_edge_dict, reduce_edge_dict = model.module.parse_edge()
        logging.info("normal edges:")
        logging.info(normal_edge_dict)
        logging.info("reduce edges:")
        logging.info(reduce_edge_dict)


def main():

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    if engine.local_rank == 0:
        logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    #data
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_data_full=ConcatDataset([train_data1,train_data2])

    valid_data = dset.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    num_train = len(train_data1)
    num_val = len(train_data2)
    if engine.local_rank == 0:
        logging.info('# images to train network: %d' % num_train)
        logging.info('# images to validate network: %d' % num_val)

    train_queue_A = engine.get_dataloader(train_data1)
    train_queue_B = engine.get_dataloader(train_data2)
    train_queue_Full = engine.get_dataloader(train_data_full)
    valid_queue = engine.get_dataloader(valid_data)

    model = Network(args.init_channels, CLASSES, args.op_search_layers, criterion)
    model.load_state_dict(torch.load("logs/search/search-imagenet_1_002_sub14_32-5448/weights_op_search.pth"))
    model.load_arch(torch.load("logs/search/search-imagenet_1_002_sub14_32-5448/arch_op_search.pth"))
    start_epoch = 30

    model = model.cuda()
    if engine.local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model = engine.data_parallel(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, Tp_Pretrain_Start, eta_min=args.learning_rate_min)

    arch_optimizer = torch.optim.Adam(model.module.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.9, 0.999),
                                      weight_decay=args.arch_weight_decay)

    for epoch in range(start_epoch,args.epochs):
        if epoch == Op_Pretrain_Start:
            model.module.phase = 'op_pretrain'
            if engine.local_rank == 0:
                logging.info("Begin operation pretrain!")
        elif epoch == Op_Search_Start:
            model.module.phase = 'op_search'
            if engine.local_rank == 0:
                logging.info("Begin operation search!")
        elif epoch == Tp_Pretrain_Start:
            model.module.__init__(args.init_channels, CLASSES, args.tp_search_layers, criterion, init_arch=False)
            model.module.phase = 'tp_pretrain'
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.learning_rate,  # use twice data to update parameters
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
            model.module.prune_model()
            arch_optimizer = torch.optim.Adam(model.module.arch_parameters(),
                                              lr=args.arch_learning_rate, betas=(0.9, 0.999),
                                              weight_decay=args.arch_weight_decay)
            model = model.cuda()
            if engine.local_rank == 0:
                logging.info("Prune model finish!")
                logging.info("Load Prune Architecture finish!")
                logging.info("Begin topology pretrain!")
        elif epoch == Tp_Search_Start:
            model.module.phase = 'tp_search'
            if engine.local_rank == 0:
                logging.info("Begin topology search!")
        else:
            pass

        if 'pretrain' in model.module.phase:
            model.module.T = 1.0
        else:
            if 'op' in model.module.phase:
                model.module.T = 1.0
            else:
                model.module.T = 10 * pow(Tp_Anneal_Rate, epoch - Tp_Search_Start)

        scheduler.step(epoch)

        lr = scheduler.get_lr()[0]

        if engine.local_rank == 0:
            logging.info('epoch:%d phase:%s lr:%e', epoch, model.module.phase, lr)
            print_genotype(model)

        # training
        if 'op' in model.module.phase:
            train_queue_A.sampler.set_epoch(epoch)
            train_queue_B.sampler.set_epoch(epoch)
            train_acc, train_obj = train_op(train_queue_A, train_queue_B, model, criterion, optimizer,arch_optimizer,engine)
        else:
            train_queue_A.sampler.set_epoch(epoch)
            train_queue_Full.sampler.set_epoch(epoch)
            train_acc, train_obj = train_tp(train_queue_A, train_queue_Full, model, criterion, optimizer, arch_optimizer,engine)

        if engine.local_rank == 0:
            logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        if engine.local_rank == 0:
            logging.info('valid_acc %f', valid_acc)
        if engine.local_rank == 0:
            utils.save_dist(model, os.path.join(args.save, 'weights_%s.pth' % model.module.phase))
            model.module.save_arch(os.path.join(args.save, 'arch_%s.pth' % model.module.phase))

    if engine.local_rank == 0:
        print_genotype(model)


def train_op(train_queue_A, train_queue_B, model, criterion, optimizer,arch_optimizer,engine):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue_A):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        # input_search, target_search = next(iter(valid_queue))
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(train_queue_B)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        if 'pretrain' not in model.module.phase:
            arch_optimizer.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            arch_optimizer.step()

        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if engine.local_rank == 0 and step % args.report_freq == 0:
            logging.info('train iter:%03d T:%4f loss:%e top1:%f top5:%f', step, model.module.T, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def train_tp(train_queue_A, train_queue_Full, model, criterion, optimizer, arch_optimizer,engine):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_queue = train_queue_A if 'pretrain' in model.module.phase else train_queue_Full

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if 'pretrain' not in model.module.phase:
            arch_optimizer.zero_grad()

        optimizer.zero_grad()

        logits = model.module.forward_tp(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        if 'pretrain' not in model.module.phase:
            arch_optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if engine.local_rank == 0 and step % args.report_freq == 0:
            logging.info('train iter:%03d T:%4f loss:%e top1:%f top5:%f', step, model.module.T, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if 'tp' in model.module.phase:
                logits = model.module.forward_tp(input)
            else:
                logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if engine.local_rank == 0 and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()


