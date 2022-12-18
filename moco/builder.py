# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MultiHeadSelfAttention(nn.Module):
    '''
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
    '''
    def __init__(self, dim_in, dim_k, dim_v, num_heads=16):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch,  nh, dk)  # (batch, nh, dk)
        k = self.linear_k(x).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch,  nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1,2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh,  dv
        att = att.transpose(1, 2).reshape(batch,  self.dim_v)  # batch, n, dim_v
        return att


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, swap=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.swap = swap

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for q, k in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            name_q, param_q = q
            name_k, param_k = k
            assert name_q == name_k
            assert param_q.shape == param_k.shape
            if len(param_q.shape)==5 and param_q.shape[3]>1:
                param_k.data = param_k.data * 0.7 + param_q.data * 0.3
#                control = torch.bernoulli(0.7 * torch.ones_like(param_q))                
#                param_k.data = param_k.data * control + param_q.data * (1-control)
            else:
                param_k.data = param_k.data * self.m + param_q.data * (1-self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_once(self, im_q, im_k, update=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # list_out = self.video_mask(im_q)
        # print(list_out.shape)
        # values, indices = list_out.topk(8, dim=None, largest=False, sorted=False)
        # for i in indices:
        #     im_q[i] = 0
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update:
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward(self, im_q, im_k):
        if self.swap:
            logits_1, labels_1 = self.forward_once(im_q, im_k)
            logits_2, labels_2 = self.forward_once(im_k, im_q, update=False)
            return logits_1, labels_1, logits_2, labels_2
        else:
            logit, label = self.forward_once(im_q, im_k)
            return logit, label





class MoCo_View(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, topk=5, mlp=False, swap=False, negative=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_View, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.topk = topk
        self.swap = swap
        self.negative = negative

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_vv", torch.randn(dim, K))
        self.queue_vv = nn.functional.normalize(self.queue_vv, dim=0)
        self.register_buffer("queue_vs", torch.randn(dim, K))
        self.queue_vs = nn.functional.normalize(self.queue_vs, dim=0)
        self.register_buffer("queue_vd", torch.randn(dim, K))
        self.queue_vd = nn.functional.normalize(self.queue_vd, dim=0)
        self.register_buffer("queue_sv", torch.randn(dim, K))
        self.queue_sv = nn.functional.normalize(self.queue_sv, dim=0)
        self.register_buffer("queue_dv", torch.randn(dim, K))
        self.queue_dv = nn.functional.normalize(self.queue_dv, dim=0)
        self.register_buffer("queue_rs", torch.randn(dim, K))
        self.queue_rs = nn.functional.normalize(self.queue_rs, dim=0)
        self.register_buffer("queue_rd", torch.randn(dim, K))
        self.queue_rd = nn.functional.normalize(self.queue_rd, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, kvv, kvs, kvd, ksv, kdv, krsv, krdv):
        # gather keys before updating queue
        kvv = concat_all_gather(kvv)
        kvs = concat_all_gather(kvs)
        kvd = concat_all_gather(kvd)
        ksv = concat_all_gather(ksv)
        kdv = concat_all_gather(kdv)
        krsv = concat_all_gather(krsv)
        krdv = concat_all_gather(krdv)

        batch_size = kvv.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_vv[:, ptr:ptr + batch_size] = kvv.T
        self.queue_vs[:, ptr:ptr + batch_size] = kvs.T
        self.queue_vd[:, ptr:ptr + batch_size] = kvd.T
        self.queue_sv[:, ptr:ptr + batch_size] = ksv.T
        self.queue_dv[:, ptr:ptr + batch_size] = kdv.T
        self.queue_rs[:, ptr:ptr + batch_size] = krsv.T
        self.queue_rd[:, ptr:ptr + batch_size] = krdv.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def calculate_logit(self, q, k, queue):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        return logits
    
    def extract_feature(self, encoder, sample, sc=None, dc=None, train=True):
        if train:
            vv, vs, vd, sv, dv = encoder(sample, sc, dc, train)
            vv = nn.functional.normalize(vv, dim=1)
            vs = nn.functional.normalize(vs, dim=1)
            vd = nn.functional.normalize(vd, dim=1)
            sv = nn.functional.normalize(sv, dim=1)
            dv = nn.functional.normalize(dv, dim=1)
            return vv, vs, vd, sv, dv
        else:
            sc, dc, sv, dv = encoder(sample, sc, dc, train)
            sv = nn.functional.normalize(sv, dim=1)
            dv = nn.functional.normalize(dv, dim=1)
            return sc, dc, sv, dv
    
    def calculate_similarity(self, logit, query, queue, similarity=None):
        if similarity is None:
            similarity = torch.einsum('nc,ck->nk', [query.detach(), queue.clone().detach()])
            similarity = nn.functional.softmax(similarity, dim=1)
        _, top_idx = torch.topk(similarity, self.topk, dim=1)
        pos_idx = torch.zeros_like(similarity)
        pos_idx.scatter_(1, top_idx, 1)
        labels = torch.cat([torch.samplerones(logit.shape[0], 1).cuda(), pos_idx], dim=1)
        return similarity, labels

    def forward_once(self, q, k, update=True):
        """
        Input:
            q: bncthw
            k: bncthw
        Output:
            logits, targets
        """

        with torch.no_grad():  # no gradient to keys
            
            qvc, qsc, qdc, qrsv, qrdv = self.extract_feature(self.encoder_k, q, train=False)
            kvc, ksc, kdc, krsv, krdv = self.extract_feature(self.encoder_k, k, train=False)

        qvv, qvs, qvd, qsv, qdv = self.extract_feature(self.encoder_q, q, qsc, qdc)  # queries: NxC

        with torch.no_grad():  # no gradient to keys
            
            if update:
                self._momentum_update_key_encoder()  # update the key encoder

            k, idx_unshuffle = self._batch_shuffle_ddp(k)
            kvv, kvs, kvd, ksv, kdv = self.extract_feature(self.encoder_k, k, ksc, kdc)  # keys: NxC
            kvv = self._batch_unshuffle_ddp(kvv, idx_unshuffle)
            kvs = self._batch_unshuffle_ddp(kvs, idx_unshuffle)
            kvd = self._batch_unshuffle_ddp(kvd, idx_unshuffle)
            ksv = self._batch_unshuffle_ddp(ksv, idx_unshuffle)
            kdv = self._batch_unshuffle_ddp(kdv, idx_unshuffle)

        vv = self.calculate_logit(qvv, kvv, self.queue_vv)
        vs = self.calculate_logit(qvs, ksv, self.queue_sv)
        vd = self.calculate_logit(qvd, kdv, self.queue_dv)
        sv = self.calculate_logit(qsv, kvs, self.queue_vs)
        dv = self.calculate_logit(qdv, kvd, self.queue_vd)
        
        similarity_vs, labels_vs = self.calculate_similarity(vs, qrsv, self.queue_rs)
        similarity_vd, labels_vd = self.calculate_similarity(vd, qrdv, self.queue_rd)
        similarity = similarity_vs + similarity_vd
        _, labels_vv = self.calculate_similarity(vv, None, None, similarity)

        # dequeue and enqueue
        self._dequeue_and_enqueue(kvv, kvs, kvd, ksv, kdv, krsv, krdv)

        return [vv, vs, vd, sv, dv], [labels_vv, labels_vs, labels_vd], [qvc, kvs+kvd]

    def forward(self, q, k):
        logit, label, caam = self.forward_once(q, k)
        return logit, label, caam

        


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
