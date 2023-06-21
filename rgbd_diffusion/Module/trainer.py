import math
import os.path as osp
from functools import partial

import torch
from diffusers import DDIMScheduler
from recon.utils import dist_info, dist_samplers, kwargs_shuffle_sampler
from rgbd_diffusion.Dataset.scannet import ScanNet
from rgbd_diffusion.Model.model import Model


class Trainer(object):
    def __init__(self):
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

        self.model = Model(self.img_size, self.fp16_mode)
        # convert batchnorm
        if self.num_rank > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        self.scaler = None
        if self.fp16_mode:
            self.scaler = torch.cuda.amp.GradScaler()

        # self.loadDataset()
        self.loadModel()
        self.loadOptimizer()
        return

    def loadDataset(self):
        dataset_create_fn = partial(ScanNet,
                                    osp.abspath("./data_file/ScanNetV2"),
                                    chunk_size=self.chunk_size,
                                    img_size=self.img_size,
                                    normalize=True,
                                    inter_mode="nearest")
        dataset_train = dataset_create_fn(
            subset_indices=self.train_split_ratio)
        dataset_test = dataset_create_fn(
            subset_indices=1.0 - self.train_split_ratio)

        samplers = dist_samplers(dataset_train, dataset_test)
        self.dataloaders = dict(train=torch.utils.data.DataLoader(
            dataset_train, **kwargs_shuffle_sampler(samplers, "train"),
            batch_size=self.batch_size_per_gpu, num_workers=4,
        ),
            val=torch.utils.data.DataLoader(
            dataset_test, **kwargs_shuffle_sampler(samplers, "val"),
            batch_size=self.batch_size_per_gpu, num_workers=2,
        ),
        )

        self.mean_std = dataset_train.normalize_mean_std.to(
            device=self.device)

        self.optimizer = None
        self.scheduler = None
        self.diffusion_scheduler = None
        return True

    def loadModel(self):
        return True

    def loadOptimizer(self):
        def learning_rate_fn(epoch):
            epoch_max = self.epoch_end
            factor_min = self.lr_final / self.lr_init
            if epoch < epoch_max:  # cosine learning rate
                return factor_min + 0.5 * (1.0 - factor_min) * \
                    (1.0 + math.cos(epoch / epoch_max * math.pi))
            else:
                return factor_min

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), weight_decay=0.01, lr=self.lr_init)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, learning_rate_fn)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False)
        return True
