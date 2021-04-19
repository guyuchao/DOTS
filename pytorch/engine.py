import os
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader
import torch.nn as nn

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)
    return tensor

class Engine(object):

    ## distribute init
    def __init__(self, args):
        self.args=args
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
        else:
            raise NotImplementedError

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
        else:
            raise NotImplementedError

    ## convert model

    def data_parallel(self, model):
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(model,device_ids=[self.local_rank],output_device=self.local_rank)
        else:
            raise NotImplementedError
        return model

    def get_dataloader(self, dataset):
        if self.distributed:
            sampler = DistributedSampler(
                dataset)
            local_bs = self.args.batch_size // self.world_size
            is_shuffle = False
            loader = DataLoader(dataset,
                   batch_size=local_bs,
                   num_workers=self.args.workers,
                   drop_last=False,
                   shuffle=is_shuffle,
                   pin_memory=True,
                   sampler=sampler)

        else:
            raise NotImplementedError

        return loader

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            raise NotImplementedError


    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            return False