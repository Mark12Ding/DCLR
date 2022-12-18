# import torch
# # video_PCA
# # video_clips = torch.randn(2, 2, 3, 12, 12)
# # number of PCA dimensions to use when reconstructing
# # B, T, C, H, W = video_clips.shape
#
#
def ni_batch(matrix):
    # matrix : B*H*W
    B, H, W = matrix.shape
    matrix = matrix.flatten(start_dim=1)
    matrix -= matrix.min(dim=-1, keepdim=True)[0]
    matrix /= (matrix.max(dim=-1, keepdim=True)[0] + 1e-8)
    return matrix.reshape(B, H, W)

def videoPCA(video_clips):
    # B C T H W
    nDims0 = 3
    B, C, T, H, W = video_clips.shape
    data = video_clips.transpose(1, 2).reshape(B, T, -1)
    data = data.permute(0, 2, 1)
    meandata = data.mean(dim=1, keepdim=True)
    data = data - meandata
    M = torch.bmm(data.permute(0, 2, 1), data)
    d, v = torch.symeig(M, eigenvectors=True)
    v = v[:, :, -nDims0:]
    Y = data.bmm(v).bmm(v.permute(0, 2, 1)) + meandata
    img_diff = (video_clips - Y.reshape(B, C, H, W, T).permute(0, 1, 4, 2, 3))
    img_diff = img_diff.abs().sum(dim=1)
    img_diff = ni_batch(img_diff.reshape(-1, H, W))
    image_batch = torchvision.utils.make_grid(img_diff.reshape(-1, 1, H, W))
    plt.imshow(image_batch.permute(1, 2, 0))
    plt.show()
    return img_diff


# import torchvision.transforms._transforms_video as transforms_video
#
import torch
import kornia
import random
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import time


class ForegroundAug(nn.Module):
    def __init__(self, gauss_size=15, H=112, W=112, device="cpu", eps=1e-8):
        super(ForegroundAug, self).__init__()
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3)).to(device)
        self.device = device
        self.h = kornia.filters.get_gaussian_kernel2d(
            kernel_size=(H, W),
            sigma=(H / 3, W / 3),
            force_even=True).to(device)
        self.h = self.h / self.h.max()
        self.eps = eps
        self.alpha = 1

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

    def getSeg(self, sampled_fg_index, sampled_bg_index, img_hsv, batch_size, T):
        # input mask:(B*T-1)*H*W, img_hsv:(B*T-1)*C*H*W
        # output soft seg:(B*T-1)*H*W
        # refine_mask = torch.zeros(mask, dtype=torch.float)
        dimH, dimS, dimV = 5, 5, 5
        H, W = img_hsv.shape[2:]
        img_hsv = img_hsv.reshape(batch_size, T, -1, H, W)  # B * T-1 * C * H* W
        h_fg = img_hsv[:, :, 0]
        s_fg = img_hsv[:, :, 1]
        v_fg = img_hsv[:, :, 2]
        hx = (s_fg * torch.cos(h_fg * 2 * np.pi) + 1) / 2
        hy = (s_fg * torch.sin(h_fg * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(v_fg * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, T, H, W
        color_map = color_map.reshape(batch_size, -1).long()
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
        return refine_mask

    def forward(self, video_clips):
        # video_clips B, C, T, H, W
        # return video_clips : B, C, T, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1)  # B, T-1, H, W
        # mask = self.ni_batch(im_diff.reshape(-1, H, W) )
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.ni_batch(mask.reshape(-1, H, W)) * self.h.unsqueeze(dim=0)  # B*T-1, H, W

        image_batch = torchvision.utils.make_grid(mask.reshape(-1, 1, H, W))
        plt.imshow(image_batch.permute(1, 2, 0))
        plt.show()

        video_clips_ = video_clips[:, :, 1:].permute(0, 2, 1, 3, 4)
        # video_clips_ = ni_batch(video_clips_.reshape(-1, H, W))
        video_clip_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B*T-1, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * T * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * T * H * W), dim=-1, largest=False)[
            1]  # shape B * K
        result_seg = self.getSeg(sampled_fg_index, sampled_bg_index, video_clip_hsv, B, T - 1)  # B*T-1*H*W
        # result_seg = self.gauss(result_seg.reshape(-1, 1, H, W))
        # *self.h.unsqueeze(0)
        # result_seg = self.gauss(result_seg.reshape(-1, 1, H, W)).reshape(-1, H, W) * self.h.unsqueeze(dim=0)
        print(result_seg.max(), self.h.max())

        # whether to add a center guassian?
        result_seg = self.ni_batch(result_seg.reshape(-1, H, W) * self.h.unsqueeze(dim=0)).reshape(B, T - 1, H, W)
        mask = torch.cat((result_seg, result_seg[:, -1:]), dim=1)  # tensor B, T, H, W
        video_fg = (video_clips * mask.reshape(-1, 1, T, H, W)).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        image_batch = torchvision.utils.make_grid(video_fg.reshape(-1, 3, H, W))
        plt.imshow(image_batch.permute(1, 2, 0))
        plt.show()


        # index = torch.randperm(B, device=self.device)
        index = torch.tensor([2, 0, 1], dtype=torch.long)
        # mask = torch.stack((mask, mask[index]), dim=0).max(dim=0)[0]
        # mid = torch.median(mask.reshape(B * T, -1), dim=-1, keepdim=True)[0]
        # mid = mid.reshape(B, T, 1, 1)
        # # mid = mask.median()
        # mask[mask >= mid] = 1
        # mask[mask < mid] = 0
        # print()
        # mask = self.gauss(mask.reshape(-1, 1, H, W)).reshape(-1, H, W) * self.h.unsqueeze(dim=0)
        # mask = self.ni_batch(mask).reshape(B, T, H, W)
        video_fuse = self.alpha * video_clips[index] * (1 - mask).unsqueeze(dim=1) + video_clips * (1-self.alpha + self.alpha * mask.unsqueeze(dim=1))

        return video_fuse

    def getSeg_v2(self, sampled_fg_index, sampled_bg_index, img_hsv, batch_size):
        # input mask:B*H*W, img_hsv:B*C*H*W
        # output soft seg:B*H*W
        # refine_mask = torch.zeros(mask, dtype=torch.float)
        dimH, dimS, dimV = 5, 5, 5
        H, W = img_hsv.shape[2:]
        img_hsv = img_hsv.reshape(batch_size, -1, H, W)  # B * C * H * W
        h_fg = img_hsv[:, 0]
        s_fg = img_hsv[:, 1]
        v_fg = img_hsv[:, 2]
        hx = (s_fg * torch.cos(h_fg * 2 * np.pi) + 1) / 2
        hy = (s_fg * torch.sin(h_fg * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(v_fg * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, T, H, W
        color_map = color_map.reshape(batch_size, -1).long()
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
        return refine_mask

    def forward_v2(self, video_clips):
        # video_clips B, C, T, H, W
        # return video_clips : B, C, T, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        # mask = self.ni_batch(im_diff.reshape(-1, H, W) )
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.ni_batch(mask.reshape(-1, H, W)) * self.h.unsqueeze(dim=0)  # B, H, W

        image_batch = torchvision.utils.make_grid(mask.reshape(-1, 1, H, W))
        plt.imshow(image_batch.permute(1, 2, 0))
        plt.show()

        video_clips_ = video_clips[:, :, 1:].permute(0, 2, 1, 3, 4).mean(dim=1) # B, C, H, W
        # video_clips_ = ni_batch(video_clips_.reshape(-1, H, W))
        video_clip_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[
            1]  # shape B * K
        result_seg = self.getSeg_v2(sampled_fg_index, sampled_bg_index, video_clip_hsv, B)  # B*H*W
        # whether to add a center guassian?
        result_seg = self.gauss(result_seg.reshape(-1, 1, H, W))
        mask = self.ni_batch(result_seg.reshape(-1, H, W)).reshape(B, H, W)
        video_fg = (video_clips * mask.reshape(-1, 1, 1, H, W)).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        image_batch = torchvision.utils.make_grid(video_fg.reshape(-1, 3, H, W))
        plt.imshow(image_batch.permute(1, 2, 0))
        plt.show()

        # index = torch.randperm(B, device=self.device)
        index = torch.tensor([2, 0, 1], dtype=torch.long)
        video_fuse = self.alpha * video_clips[index] * (1 - mask).reshape(-1, 1, 1, H, W) + video_clips * (1-self.alpha + self.alpha * mask.reshape(-1, 1, 1, H, W))

        return video_fuse

import torchvision.transforms._transforms_video as transforms_video
#
path = "Z37MH-38BB0_000008_000018.mp4"
video = torchvision.io.read_video(path, start_pts=2, end_pts=8, pts_unit='sec')
video_1 = video[0][:16]
path = "Zj67gmFupiY_000036_000046.mp4"
video = torchvision.io.read_video(path, start_pts=4, end_pts=5, pts_unit='sec')
video_2 = video[0][:16]
path = "GoohUhM-raM_000008_000018.mp4"
video = torchvision.io.read_video(path, start_pts=2, end_pts=10, pts_unit='sec')
video_3 = video[0][:16]
video_augmentation = transforms.Compose(
    [
        transforms_video.ToTensorVideo(),
        transforms_video.RandomResizedCropVideo(112, (0.8, 1))
        # moco.loader.MoCoAugment(args.crop_size, args.fa)
    ]
)
video_1 = video_augmentation(video_1)
video_2 = video_augmentation(video_2)
video_3 = video_augmentation(video_3)

video_clips = torch.stack([video_1, video_2, video_3], dim=0)
print(video_clips.shape, "raw clips shape")
# videoPCA(video_clips)
vis = video_clips.permute(0, 2, 1, 3, 4).reshape(-1, 3, 112, 112)
# print(vis.shape)
image_batch = torchvision.utils.make_grid(vis)
plt.imshow(image_batch.permute(1, 2, 0))
plt.show()
FA = ForegroundAug()
out = FA.forward_v2(video_clips).permute(0, 2, 1, 3, 4).reshape(-1, 3, 112, 112)
image_batch = torchvision.utils.make_grid(out)
plt.imshow(image_batch.permute(1, 2, 0))
plt.show()
