# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import torch.nn as nn
import kornia
import torchvision.transforms as transforms
import numpy as np
from kornia.augmentation.container import VideoSequential
import time

class ForegroundAug(nn.Module):
    def __init__(self, crop_size=112, beta=0.5, device="cpu", eps=1e-8):
        super(ForegroundAug, self).__init__()
        self.crop_size = crop_size
        gauss_size = int(0.1*crop_size)//2*2+1
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3)).to(device)
        self.device = device

        ##### For Gauss mask #######
        self.h = kornia.filters.get_gaussian_kernel2d(
            kernel_size=(crop_size, crop_size),
            sigma=(crop_size / 3, crop_size / 3),
            force_even=True).to(device)
        self.h = self.h / self.h.max()
        self.eps = eps
        self.alpha = 1
        self.beta = beta # control the portion of foreground
        self.grid = self.init_grid_mask()

    def init_grid_mask(self):
        grid_mask = torch.zeros(16, self.crop_size, self.crop_size, device=self.device)
        for i in range(16):
            h = i // 4
            w = i % 4
            grid_mask[i, int(self.crop_size/4*h):int(self.crop_size/4*(h+1)), int(self.crop_size/4*w):int(self.crop_size/4*(w+1))] = 1
        return grid_mask

    def ni_batch(self, matrix):
        # matrix : B*H*W
        B, H, W = matrix.shape
        matrix = matrix.flatten(start_dim=1)
        matrix -= matrix.min(dim=-1, keepdim=True)[0]
        matrix /= (matrix.max(dim=-1, keepdim=True)[0] + self.eps)
        return matrix.reshape(B, H, W)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target
    
    def getSeg(self, mask, video_clips):
        # input mask:B*H*W, video_clips:B, C, T, H, W
        # output soft seg:B*H*W
        # refine_mask = torch.zeros(mask, dtype=torch.float)
        B, C, T, H, W = video_clips.shape
        video_clips_ = video_clips[:, :, 1:].permute(0, 2, 1, 3, 4).mean(dim=1) # B, C, H, W
        # video_clips_ = ni_batch(video_clips_.reshape(-1, H, W))
        img_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.5 * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[
            1]  # shape B * K
        
        dimH, dimS, dimV = 10, 10, 10
        H, W = img_hsv.shape[2:]
        img_hsv = img_hsv.reshape(B, -1, H, W)  # B * C * H * W
        h_fg = img_hsv[:, 0]
        s_fg = img_hsv[:, 1]
        v_fg = img_hsv[:, 2]
        hx = (s_fg * torch.cos(h_fg * 2 * np.pi) + 1) / 2
        hy = (s_fg * torch.sin(h_fg * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(v_fg * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, T, H, W
        color_map = color_map.reshape(B, -1).long()
        col_fg = color_map.gather(index=sampled_fg_index, dim=-1)  # B * K
        col_bg = color_map.gather(index=sampled_bg_index, dim=-1)  # B * K
        # dict_fg = torch.zeros(B, dimH * dimS * dimV, dtype=torch.float)
        dict_fg = self.batched_bincount(col_fg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        # dict_fg = torch.zeros(B, dimH * dimS * dimV, dtype=torch.float)
        dict_bg = self.batched_bincount(col_bg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_fg = dict_fg.float()
        dict_bg = dict_bg.float() + 1
        dict_fg /= (dict_fg.sum(dim=-1, keepdim=True) + self.eps)
        dict_bg /= (dict_bg.sum(dim=-1, keepdim=True) + self.eps)
        # col_fg # B * T * H * W
        # dict_fg # B * (dimH * dimS * dimV)
        pr_fg = dict_fg.gather(dim=1, index=color_map)
        pr_bg = dict_bg.gather(dim=1, index=color_map)
        refine_mask = pr_fg / (pr_bg + pr_fg)

        mask = self.gauss(refine_mask.reshape(-1, 1, H, W))
        mask = self.ni_batch(mask.reshape(-1, H, W))
        num_fg = int(self.beta * H * W)
        sampled_index = torch.topk(mask.reshape(B, -1), k=num_fg, dim=-1)[1]
        # print(sampled_index.shape) 
        mask = torch.zeros_like(mask).reshape(B, -1)
        b_index = torch.LongTensor([[i]*num_fg for i in range(B)])
        mask[b_index.view(-1),sampled_index.view(-1)] = 1
        # mid = mask.reshape(B, -1).median(dim=-1)[0].reshape(B, 1, 1) * torch.ones_like(mask)
        # mask[mask>=mid] = 1
        # mask[mask<mid] = 0
        return mask.reshape(B, H, W)

    def getGrid(self, mask):
        B, H, W = mask.shape
        # activation = torch.zeros(B, 16, dtype=mask.dtype, device=mask.device)
        activation = mask.reshape(B, -1).matmul(self.grid.reshape(16, -1).permute(1,0)) # B, 16
        fg_index = torch.topk(activation, k=8, dim=-1)[1] # B * 8
        mask = self.grid[fg_index.view(-1)].reshape(B, 8, H, W).sum(dim=1)
        return mask

    def forward(self, video_clips, grid=True):
        # video_clips B, C, T, H, W
        # return video_clips : B, C, T, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        # mask = self.ni_batch(im_diff.reshape(-1, H, W) )
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.ni_batch(mask.reshape(-1, H, W))  # B, H, W
        mask = self.getGrid(mask) if grid else self.getSeg(mask, video_clips)
        # video_fg = (video_clips * mask.reshape(-1, 1, 1, H, W)).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        index = torch.randperm(B, device=self.device)
        video_fuse = self.alpha * video_clips[index] * (1 - mask).reshape(-1, 1, 1, H, W) + video_clips * (1-self.alpha + self.alpha * mask.reshape(-1, 1, 1, H, W))
        return video_fuse

    def gaussion_only(self, video_clips):
        # video_clips B, C, T, H, W
        B, C, T, H, W = video_clips.shape
        index = torch.randperm(B, device=self.device)
        video_fuse = video_clips[index] * (1 - self.h).reshape(-1, 1, 1, H, W) + video_clips * self.h.reshape(-1, 1, 1, H, W)
        return video_fuse

    # def BE(self, video_clips):
    #     B, C, T, H, W = video_clips.shape
    #     loss_prob = np.random.random() * 0.3
    #     img_index = np.random.randint(T)
    #     video_fuse = (1-loss_prob) * video_clips + loss_prob * video_clips[:,:,img_index].unsqueeze(2)
    #     return video_fuse

### stronger augmentation
def MocoAugment_GPU(args):
    crop_size = args.crop_size
    radius = int(0.1*crop_size)//2*2+1
    sigma = random.uniform(0.1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        # kornia.augmentation.ColorJitter(para, para, para, para),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.augmentation.RandomGaussianBlur((radius, radius), (sigma, sigma), p=0.5),
        normalize_video,
        data_format="BCTHW",
        same_on_frame=True)
    return aug_list


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.], img_size=112):
        self.sigma = sigma
        self.radius = int(0.1*img_size)//2*2+1

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gauss = kornia.filters.GaussianBlur2d((self.radius, self.radius), (sigma, sigma))
        return gauss(x)


class MoCoAugment(object):

    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment = transforms.Compose(
            [
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.4),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


class MoCoAugmentV2(object):
    def __init__(self, crop_size):
        print('crop size in augmentv2 is {}'.format(crop_size))
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment_v2 = transforms.Compose(
            [
                transforms.RandomApply([
                    kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.], crop_size)], p=0.5),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video,
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


class RandomTwoClipSampler(Sampler):
    """
    Samples two clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select two clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 2:
                sampled = [s, s]
            else:
                sampled = torch.randperm(length)[:2] + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        # shuffle all clips randomly
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)



class OrderTwoClipSampler(Sampler):
    """
    Samples two clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select two clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 2:
                sampled = [s, s, s+length]
            else:
                sampled = torch.randperm(length)[:2] + s
                sampled = sampled.tolist()
                sampled.append(s+length)
            s += length
            idxs.append(sampled)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)



class DummyAudioTransform(object):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)