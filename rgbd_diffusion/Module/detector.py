import os

from diffusers import DDIMScheduler
import torch
from einops import rearrange
from recon.utils import dist_info
from rgbd_diffusion.Dataset.scannet_scene import ScanNetScene
from rgbd_diffusion.Model.model import Model


class Detector(object):
    def __init__(self, model_file_path):
        self.chunk_size = 8
        self.img_size = 128
        self.train_split_ratio = 0.8016702610930115

        self.fp16_mode = True
        self.lr_init = 1e-4
        self.lr_final = 1e-6
        self.batch_size_per_gpu = 40
        self.seed = 3407

        # training
        self.epoch_end = 300

        # routine
        self.print_every = 10
        self.checkpoint_latest_every = 1000
        self.checkpoint_every = 8000
        self.validate_every = 8000
        self.visualize_every = -1

        # save
        self.model_selection_metric = "loss"
        self.model_selection_mode = "minimize"
        self.backup_lst = ["dataset/ScanNet"]

        self.rank, self.num_rank, self.device = dist_info()
        self.batch_size = self.batch_size_per_gpu * self.num_rank
        torch.manual_seed(self.seed)

        self.train_loader = None
        self.val_loader = None
        self.mean_std = None

        self.model = Model(self.img_size, self.fp16_mode)
        # convert batchnorm
        if self.num_rank > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        self.scaler = None
        if self.fp16_mode:
            self.scaler = torch.cuda.amp.GradScaler()

        # run.sample
        self.guidance = 1.0
        self.inpainting = True
        self.save_gt = False
        self.strict = True
        self.num_steps = 50
        self.voxel_size = 0.02
        self.min_views = 50
        self.ind_scenes = [0]
        self.task = 0.1
        self.sample_seed = 0
        self.traj = ''
        self.interactive = True
        self.suffix = ''

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False)

        self.loadModel(model_file_path)
        return

    @torch.no_grad()
    def sample(self, scene_mesh, cam_curr, mean, std, seed=3407):
        """ Sample A Novel View for One Scene
        Args:
            scene_mesh (tuple): tuple of `vertices`, `faces`, `colors`
            cam_curr (tuple): camera intrinsic and pose matrices, (3, 3/4)
        """
        # sample noise
        def kwargs_rand(seed, device=self.device): return dict(
            generator=torch.Generator(device).manual_seed(seed), device=device)
        z_t = torch.randn(
            [1, 4, self.img_size, self.img_size], **kwargs_rand(seed))
        # compute the known part by rendering the mesh onto the current view
        try:
            rgb_out, d_out, vis_out = self.model.render_many(
                (*scene_mesh,
                 torch.tensor([[0, len(scene_mesh[1]),
                                0, len(scene_mesh[0])]])),
                (cam_curr[0][None, ...], cam_curr[1][None, ...]),
                res=self.img_size,
            )
        except Exception as e:
            if self.strict:
                raise e
            print(f"[WARNING] cannot render, because of: {e}")
            rgb_out = torch.zeros(
                [1, 3, self.img_size, self.img_size], device=self.device)
            d_out = torch.zeros(
                [1,    self.img_size, self.img_size], device=self.device)
            vis_out = torch.zeros(
                [1,    self.img_size, self.img_size], device=self.device)
        rgbd_out = torch.cat([rgb_out, d_out[:, None]], dim=1)  # (1, 4, H, W)
        known_part = (rgbd_out - mean[None, :, None, None]) / \
            std[None, :, None, None]  # (1, 4, H, W)
        known_part_msk = vis_out[:, None].float()  # (1, 1, H, W)
        #
        known_part = rearrange(known_part, "B C H W -> B H W C")
        known_part[~known_part_msk[0].bool()] = self.model.no_pixel
        known_part = rearrange(known_part, "B H W C -> B C H W")
        #
        self.diffusion_scheduler.set_timesteps(self.num_steps)
        #
        time_step_lst = self.diffusion_scheduler.timesteps
        num_steps_train = self.diffusion_scheduler.config.num_train_timesteps
        assert self.num_steps == self.diffusion_scheduler.num_inference_steps \
            == len(time_step_lst)
        seeds = torch.randint(0, int(1e6), size=(
            self.num_steps - 1, ), **kwargs_rand(seed + 1, device="cpu"))
        #
        self.model.eval()
        for i, t in enumerate(time_step_lst):
            rgbd_in = torch.cat([known_part, z_t], dim=1)  # (1, 8, H, W)
            pred_noise = self.sampling_forward_fn(
                rgbd_in, t.to(device=self.device))
            z_t = self.diffusion_scheduler.step(
                pred_noise, t.to(device=self.device), z_t).prev_sample
            # add noise to masking region
            if i < self.num_steps - 1:
                prev_timestep = t - num_steps_train // self.num_steps
                z_t_known = self.diffusion_scheduler.add_noise(
                    known_part,
                    noise=torch.randn(known_part.shape, **
                                      kwargs_rand(seeds[i].item())),
                    timesteps=prev_timestep.reshape([1]),
                )
            else:
                # the last step, use the previous RGBD, don't add noise anymore
                z_t_known = known_part
            # do masking
            if self.inpainting:
                z_t = z_t * (1.0 - known_part_msk) + \
                    z_t_known * known_part_msk
        # reshape to (H, W, C)
        return rearrange(z_t, "() C H W -> H W C"), known_part_msk[0, 0].bool()

    @torch.no_grad()
    # forward func only for classifier-free sampling
    def sampling_forward_fn(self, rgbd, t):
        t = t.reshape([1])
        # forward
        with torch.cuda.amp.autocast(enabled=self.fp16_mode):
            if self.guidance == 0.0:
                rgbd[:, :4, ...] = self.model.no_pixel[None, :, None, None]
                pred = self.model.unet(rgbd, t)
            elif self.guidance == 1.0:
                pred = self.model.unet(rgbd, t)
            else:
                pred_cond = self.model.unet(rgbd, t)
                rgbd[:, :4, ...] = self.model.no_pixel[None, :, None, None]
                pred_uncond = self.model.unet(rgbd, t)
                pred = pred_uncond + self.guidance * (pred_cond - pred_uncond)
        return pred.float()

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            return False

        torch.load(model_file_path)

        self.model.to(device=self.device)
        return True

    def detect(self):
        return True

    def detectDataset(self, dataset_folder_path):
        dataset = ScanNetScene(
            path=dataset_folder_path,
            num_views_sample=None,
            max_len_seq=None,
            img_size=self.img_size,
            normalize=True,
            inter_mode="nearest",
            subset_indices=1.0 - self.train_split_ratio,
            reverse_frames=False,
        )

        mean, std = dataset.normalize_mean_std.to(
            device=self.device).unbind(dim=1)
        return True
