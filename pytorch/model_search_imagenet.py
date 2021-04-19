import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np
from itertools import combinations
import torch


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights_edge):

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        offset_tp = 0
        for i in range(self._steps):
            if weights_edge is None:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
            else:
                comb = list(combinations(list(range(i + 2)), 2))
                # print("combination:%d"%i,comb)
                offset_tp_end = offset_tp + Cal_Combination_Numbers(2 + i, 2)
                # print("begin,end:",offset_tp,offset_tp_end)
                w_tp = weights_edge[offset_tp:offset_tp_end]
                # print("weight_edge_shape:",weights_edge.shape)
                w_tp_edge = torch.zeros(len(states), device=weights_edge.device, dtype=weights_edge.dtype)
                # print("w_tp_edge_shape:",w_tp_edge.shape)

                for w, edges in zip(w_tp, comb):
                    w_tp_edge[edges[0]] += w
                    w_tp_edge[edges[1]] += w
                # print("w_tp_edge:",w_tp_edge)

                s = sum(w_tp_edge[j] * self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                offset_tp = offset_tp_end
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


def Cal_Combination_Numbers(n, r):
    '''
    C_{n}^{r}
    :param m:
    :param n:
    :return:
    '''
    return int(np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r)))


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, init_arch=True):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        self.phase = 'op_pretrain'
        self.T = 1.0

        C_curr = stem_multiplier * C
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()

        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(C_prev, num_classes)
        if init_arch:
            self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def parse_edge(self):
        betas_normal = self.betas_normal
        betas_reduce = self.betas_reduce
        normal_edge_dict = {}
        reduce_edge_dict = {}
        for step in range(self._steps):
            normal_edge_dict['normal_node1'] = list(F.softmax(betas_normal[0:1], dim=-1).detach().cpu().numpy())
            normal_edge_dict['normal_node2'] = list(F.softmax(betas_normal[1:4], dim=-1).detach().cpu().numpy())
            normal_edge_dict['normal_node3'] = list(F.softmax(betas_normal[4:10], dim=-1).detach().cpu().numpy())
            normal_edge_dict['normal_node4'] = list(F.softmax(betas_normal[10:20], dim=-1).detach().cpu().numpy())
        for step in range(self._steps):
            reduce_edge_dict['reduce_node1'] = list(F.softmax(betas_reduce[0:1], dim=-1).detach().cpu().numpy())
            reduce_edge_dict['reduce_node2'] = list(F.softmax(betas_reduce[1:4], dim=-1).detach().cpu().numpy())
            reduce_edge_dict['reduce_node3'] = list(F.softmax(betas_reduce[4:10], dim=-1).detach().cpu().numpy())
            reduce_edge_dict['reduce_node4'] = list(F.softmax(betas_reduce[10:20], dim=-1).detach().cpu().numpy())
        return normal_edge_dict, reduce_edge_dict

    def prune_model(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.bool_mask_normal = torch.zeros(k, num_ops)
        self.bool_mask_reduce = torch.zeros(k, num_ops)

        alphas_normal_non_params = self.alphas_normal_non_params
        alphas_normal_params = self.alphas_normal_params
        alphas_reduce_non_params = self.alphas_reduce_non_params
        alphas_reduce_params = self.alphas_reduce_params

        for idx, weight in enumerate(alphas_normal_params):
            self.bool_mask_normal[idx, 4 + torch.argmax(weight)] = 1
        for idx, weight in enumerate(alphas_normal_non_params):
            self.bool_mask_normal[idx, torch.argmax(weight)] = 1
        for idx, weight in enumerate(alphas_reduce_params):
            self.bool_mask_reduce[idx, 4 + torch.argmax(weight)] = 1
        for idx, weight in enumerate(alphas_reduce_non_params):
            self.bool_mask_reduce[idx, torch.argmax(weight)] = 1

        self.bool_mask_normal = self.bool_mask_normal.cuda()
        self.bool_mask_reduce = self.bool_mask_reduce.cuda()

        # init prune model
        self.alphas_normal_balance = Variable(1e-3 * torch.zeros(k, 2).cuda(), requires_grad=True)
        self.alphas_reduce_balance = Variable(1e-3 * torch.zeros(k, 2).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.zeros(
            Cal_Combination_Numbers(2, 2) + Cal_Combination_Numbers(3, 2) + Cal_Combination_Numbers(4,
                                                                                                    2) + Cal_Combination_Numbers(
                5, 2)).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.zeros(
            Cal_Combination_Numbers(2, 2) + Cal_Combination_Numbers(3, 2) + Cal_Combination_Numbers(4,
                                                                                                    2) + Cal_Combination_Numbers(
                5, 2)).cuda(), requires_grad=True)
        del self.alphas_normal_non_params
        del self.alphas_normal_params
        del self.alphas_reduce_non_params
        del self.alphas_reduce_params
        self._arch_parameters = [
            self.alphas_normal_balance,
            self.alphas_reduce_balance,
            self.betas_normal,
            self.betas_reduce,
        ]

    def forward(self, input):

        assert 'op' in self.phase, "error"

        weights_reduce_non_params = F.softmax(self.alphas_reduce_non_params / self.T, dim=-1)
        weights_reduce_params = F.softmax(self.alphas_reduce_params / self.T, dim=-1)
        weights_reduce = torch.cat([weights_reduce_non_params, weights_reduce_params], -1)

        weights_normal_non_params = F.softmax(self.alphas_normal_non_params / self.T, dim=-1)
        weights_normal_params = F.softmax(self.alphas_normal_params / self.T, dim=-1)
        weights_normal = torch.cat([weights_normal_non_params, weights_normal_params], -1)

        s0 = self.stem0(input)
        s1 = self.stem1(s0)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
                weights_edge = None
            else:
                weights = weights_normal
                weights_edge = None
            s0, s1 = s1, cell(s0, s1, weights, weights_edge)

        s1 = self.lastact(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def forward_tp(self, input):
        assert 'tp' in self.phase, "error"

        weights_reduce_balance = F.softmax(self.alphas_reduce_balance / (1e-3 * self.T), dim=-1)  # 1e-3 or 1e-1
        weights_reduce = torch.zeros_like(self.bool_mask_reduce)
        weights_reduce[:, :4] = self.bool_mask_reduce[:, :4] * weights_reduce_balance[:, :1]
        weights_reduce[:, 4:] = self.bool_mask_reduce[:, 4:] * weights_reduce_balance[:, 1:]

        weights_normal_balance = F.softmax(self.alphas_normal_balance / (1e-3 * self.T), dim=-1)
        weights_normal = torch.zeros_like(self.bool_mask_normal)
        weights_normal[:, :4] = self.bool_mask_normal[:, :4] * weights_normal_balance[:, :1]
        weights_normal[:, 4:] = self.bool_mask_normal[:, 4:] * weights_normal_balance[:, 1:]

        s0 = self.stem0(input)
        s1 = self.stem1(s0)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce

                start = 1
                weights_edge = F.softmax(self.betas_reduce[0:Cal_Combination_Numbers(2, 2)] / self.T, dim=-1)
                for i in range(1, self._steps):
                    end = start + Cal_Combination_Numbers(2 + i, 2)
                    tw2 = F.softmax(self.betas_reduce[start:end] / self.T, dim=-1)
                    start = end
                    weights_edge = torch.cat([weights_edge, tw2], dim=0)

            else:
                weights = weights_normal

                start = 1
                weights_edge = F.softmax(self.betas_normal[0:Cal_Combination_Numbers(2, 2)] / self.T, dim=-1)
                for i in range(1, self._steps):
                    end = start + Cal_Combination_Numbers(2 + i, 2)
                    tw2 = F.softmax(self.betas_normal[start:end] / self.T, dim=-1)
                    start = end
                    weights_edge = torch.cat([weights_edge, tw2], dim=0)

            s0, s1 = s1, cell(s0, s1, weights, weights_edge)

        s1 = self.lastact(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def load_arch(self, state_dict):
        if 'alphas_normal_non_params' in state_dict:
            self.alphas_normal_non_params.data.copy_(state_dict["alphas_normal_non_params"])
        if 'alphas_normal_params' in state_dict:
            self.alphas_normal_params.data.copy_(state_dict["alphas_normal_params"])
        if 'alphas_reduce_params' in state_dict:
            self.alphas_reduce_params.data.copy_(state_dict["alphas_reduce_params"])
        if 'alphas_reduce_non_params' in state_dict:
            self.alphas_reduce_non_params.data.copy_(state_dict["alphas_reduce_non_params"])
        if 'alphas_normal_balance' in state_dict:
            self.alphas_normal_balance.data.copy_(state_dict["alphas_normal_balance"])
        if 'alphas_reduce_balance' in state_dict:
            self.alphas_reduce_balance.data.copy_(state_dict["alphas_reduce_balance"])
        if 'betas_normal' in state_dict:
            self.betas_normal.data.copy_(state_dict["betas_normal"])
        if 'betas_reduce' in state_dict:
            self.betas_reduce.data.copy_(state_dict["betas_reduce"])
        if 'bool_mask_normal' in state_dict:
            self.bool_mask_normal.data.copy_(state_dict["bool_mask_normal"])
        if 'bool_mask_reduce' in state_dict:
            self.bool_mask_reduce.data.copy_(state_dict["bool_mask_reduce"])

    def _loss(self, input, target):
        if 'tp' in self.phase:
            logits_onehot = self.forward_oh(input)
            return self._criterion(logits_onehot, target)
        else:
            logits = self(input)
            return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # hardcode params [conv3*3, conv5*5, dil3*3, dil5*5] no_params [zero, maxpool, avgpool, identity]

        # group weight
        self.alphas_normal_non_params = Variable(1e-3 * torch.zeros(k, 4).cuda(), requires_grad=True)
        self.alphas_normal_params = Variable(1e-3 * torch.zeros(k, 4).cuda(), requires_grad=True)

        self.alphas_reduce_non_params = Variable(1e-3 * torch.zeros(k, 4).cuda(), requires_grad=True)
        self.alphas_reduce_params = Variable(1e-3 * torch.zeros(k, 4).cuda(), requires_grad=True)

        self._arch_parameters = [
            self.alphas_normal_non_params,
            self.alphas_normal_params,
            self.alphas_reduce_non_params,
            self.alphas_reduce_params
        ]

    def save_arch(self, path):
        if 'op' in self.phase:
            state_dict = {
                "alphas_normal_non_params": self.alphas_normal_non_params,
                "alphas_normal_params": self.alphas_normal_params,
                "alphas_reduce_non_params": self.alphas_reduce_non_params,
                "alphas_reduce_params": self.alphas_reduce_params
            }
        else:
            state_dict = {
                "bool_mask_normal": self.bool_mask_normal,
                "bool_mask_reduce": self.bool_mask_reduce,
                "alphas_normal_balance": self.alphas_normal_balance,
                "alphas_reduce_balance": self.alphas_reduce_balance,
                "betas_normal": self.betas_normal,
                "betas_reduce": self.betas_reduce,
            }
        torch.save(state_dict, path)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse_op(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))  # geno item: (operation, node idx)
                start = end
                n += 1
            return gene

        def _parse_tp(weights, weights2):
            gene = []
            n = 2
            start_op = 0
            start_tp = 0
            for i in range(self._steps):
                end_op = start_op + n
                end_tp = start_tp + Cal_Combination_Numbers(2 + i, 2)
                W = weights[start_op:end_op].copy()
                W2 = weights2[start_tp:end_tp].copy()
                edges = list(combinations(list(range(2 + i)), 2))[W2.argmax()]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start_tp = end_tp
                start_op = end_op
                n += 1
            return gene

        print(self.phase)
        if 'op' in self.phase:
            weights_reduce_non_params = F.softmax(self.alphas_reduce_non_params, dim=-1).cpu()
            weights_reduce_params = F.softmax(self.alphas_reduce_params, dim=-1).cpu()
            weights_reduce = torch.cat([weights_reduce_non_params, weights_reduce_params], -1)

            weights_normal_non_params = F.softmax(self.alphas_normal_non_params, dim=-1).cpu()
            weights_normal_params = F.softmax(self.alphas_normal_params, dim=-1).cpu()
            weights_normal = torch.cat([weights_normal_non_params, weights_normal_params], -1)

            gene_normal = _parse_op(weights_normal.data.cpu().numpy())
            gene_reduce = _parse_op(weights_reduce.data.cpu().numpy())

        else:
            start = 1
            weightsr2 = F.softmax(self.betas_reduce[0:Cal_Combination_Numbers(2, 2)], dim=-1)
            weightsn2 = F.softmax(self.betas_normal[0:Cal_Combination_Numbers(2, 2)], dim=-1)
            for i in range(1, self._steps):
                end = start + Cal_Combination_Numbers(2 + i, 2)
                tw2 = F.softmax(self.betas_reduce[start:end] / self.T, dim=-1)
                tn2 = F.softmax(self.betas_normal[start:end] / self.T, dim=-1)
                start = end
                weightsr2 = torch.cat([weightsr2, tw2], dim=0)
                weightsn2 = torch.cat([weightsn2, tn2], dim=0)

            weights_reduce_balance = F.softmax(self.alphas_reduce_balance, dim=-1)
            weights_reduce = torch.zeros_like(self.bool_mask_reduce)
            weights_reduce[:, :4] = self.bool_mask_reduce[:, :4] * weights_reduce_balance[:, :1]
            weights_reduce[:, 4:] = self.bool_mask_reduce[:, 4:] * weights_reduce_balance[:, 1:]

            weights_normal_balance = F.softmax(self.alphas_normal_balance, dim=-1)
            weights_normal = torch.zeros_like(self.bool_mask_normal)
            weights_normal[:, :4] = self.bool_mask_normal[:, :4] * weights_normal_balance[:, :1]
            weights_normal[:, 4:] = self.bool_mask_normal[:, 4:] * weights_normal_balance[:, 1:]

            gene_normal = _parse_tp(weights_normal.data.cpu().numpy(), weightsn2.data.cpu().numpy())
            gene_reduce = _parse_tp(weights_reduce.data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype, weights_normal, weights_reduce

