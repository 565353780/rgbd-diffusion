import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange
from rgbd_diffusion.Module.render import Render

sys.path.append('./third_party/latent-diffusion/')


def get_unet(img_size, fp16_mode):
    from ldm.modules.diffusionmodules.openaimodel import UNetModel
    unet = UNetModel(image_size=img_size,
                     in_channels=8, out_channels=4,
                     model_channels=128,  # the base channel (smallest)
                     channel_mult=[1, 2, 3, 3, 4, 4],
                     num_res_blocks=2,
                     num_head_channels=32,
                     # down 1    2    4         8         16        32
                     # res  128  64   32        16        8         4
                     # chan 128  256  384       384       512       512
                     # type conv conv conv+attn conv+attn conv+attn conv+attn
                     attention_resolutions=[4, 8, 16, 32],
                     use_checkpoint=True,
                     use_fp16=fp16_mode,
                     )

    # use num_groups==1, to avoid color shift problem
    for name, module in unet.named_modules():
        if isinstance(module, nn.GroupNorm):
            module.num_groups = 1
            print(f"convert GN to LN for module: {name}")
    return unet


class Model(nn.Module, Render):
    def __init__(self, img_size, fp16_mode):
        nn.Module.__init__(self)
        Render   .__init__(self)
        self.unet = get_unet(img_size, fp16_mode)
        # dummy params
        self.no_pixel = nn.Parameter(torch.zeros(4))
        self.img_size = img_size
        return

    def render_views(self,
                     # source views (e.g. previous views), (B, N, C, H, W)
                     rgbd_src,
                     # camera params of source views, (B, N, 3, 3/4)
                     c2w_src,
                     # project onto this camera view, (B, 3, 3/4)
                     c2w_dst,
                     mean_std
                     ):
        B, N = rgbd_src.shape[:2]
        rgbd_src = rgbd_src.flatten(0, 1)  # (B*N, C, H, W)
        # undo normalization
        mean, std = mean_std.unbind(dim=1)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        rgbd_src = rgbd_src * std + mean
        #
        c2w_src = tuple(mat.flatten(0, 1) for mat in c2w_src)
        vertices, faces, attrs, ind_group_src = self.meshing_many(
            rgbd_src, *c2w_src)
        #
        ind_group_src = ind_group_src.reshape(B, N, 4)
        ind_group_dst = ind_group_src.new_zeros([B, 4])
        ind_group_dst[:, 0] = ind_group_src[:, 0, 0]            # face start
        ind_group_dst[:, 1] = ind_group_src[:, :, 1].sum(dim=1)  # face count
        ind_group_dst[:, 2] = ind_group_src[:, 0, 2]            # vert start
        ind_group_dst[:, 3] = ind_group_src[:, :, 3].sum(dim=1)  # vert count
        #
        img_attr, img_dep, mask_out = self.render_many(
            (vertices, faces, attrs, ind_group_dst),
            c2w_dst, res=self.img_size,
        )
        rgbd_out = torch.cat(
            [img_attr, img_dep[:, None, ...]], dim=1)  # (B, C, H, W)
        # redo normalization
        rgbd_out = (rgbd_out - mean) / std
        # some pixels are empty because they haven't been projected onto
        rgbd_out = rearrange(rgbd_out, "B C H W -> B H W C")
        rgbd_out[~mask_out] = self.no_pixel
        rgbd_out = rearrange(rgbd_out, "B H W C -> B C H W")
        # (B, C, H, W) / (B, H, W)
        return SimpleNamespace(rgbd=rgbd_out, mask=mask_out)

    def forward(self, rgbd, cam, t, mean_std):
        """
        Args:
            rgbd.shape == (B, N, C, H, W)
            cam (tuple): include intrinsic and pose matrices,
            shape == (B, N, 3, 3) / (B, N, 3, 4)

            NOTE
            only curr view (i.e. rgbd[:, -1]) is noised
            N == num of views (prev + curr)
            invalid depth is zero, then will be trimmed
        """
        B, N = rgbd.shape[:2]
        assert N >= 2, "at least one previous view is provided. \
        if there is no previous views, please provide zero"
        camint, camext = cam
        rgbd_render = self.render_views(
            rgbd[:, :-1],
            (camint[:, :-1], camext[:, :-1]),
            (camint[:,  -1], camext[:,  -1]),
            mean_std).rgbd  # (B, C, H, W)
        #
        unet_in = torch.cat([rgbd_render, rgbd[:, -1]],
                            dim=1)  # (B, 4 + 4, H, W)
        # (B, C, H, W)
        return self.unet(unet_in, t)
