import math
import os
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from recon.utils import dist_info, dist_samplers, kwargs_shuffle_sampler
from rgbd_diffusion.Dataset.scannet import ScanNet
from rgbd_diffusion.Method.augment import data_augmentation
from rgbd_diffusion.Method.device import move_to
from rgbd_diffusion.Method.loss import combine_loss
from rgbd_diffusion.Method.path import createFileFolder, removeFile, renameFile
from rgbd_diffusion.Method.time import getCurrentTime
from rgbd_diffusion.Model.model import Model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer(object):
    def __init__(self, dataset_folder_path, model_file_path,
                 resume_model_only=False):
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

        self.optimizer = None
        self.scheduler = None
        self.diffusion_scheduler = None

        self.model = Model(self.img_size, self.fp16_mode)
        # convert batchnorm
        if self.num_rank > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        self.scaler = None
        if self.fp16_mode:
            self.scaler = torch.cuda.amp.GradScaler()

        self.step = 0
        self.eval_step = 0
        self.loss_min = float('inf')
        self.eval_loss_min = float('inf')
        self.log_folder_name = getCurrentTime()

        self.summary_writer = None

        self.loadDataset(dataset_folder_path)
        self.loadOptimizer()
        self.loadModel(model_file_path, resume_model_only)
        return

    def loadDataset(self, dataset_folder_path):
        dataset_create_fn = partial(ScanNet,
                                    dataset_folder_path,
                                    chunk_size=self.chunk_size,
                                    img_size=self.img_size,
                                    normalize=True,
                                    inter_mode="nearest")
        dataset_train = dataset_create_fn(
            subset_indices=self.train_split_ratio)
        dataset_test = dataset_create_fn(
            subset_indices=1.0 - self.train_split_ratio)

        samplers = dist_samplers(dataset_train, dataset_test)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, **kwargs_shuffle_sampler(samplers, "train"),
            batch_size=self.batch_size_per_gpu, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(
            dataset_test, **kwargs_shuffle_sampler(samplers, "val"),
            batch_size=self.batch_size_per_gpu, num_workers=2)

        self.mean_std = dataset_train.normalize_mean_std.to(
            device=self.device)
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

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter(f"./logs/{self.log_folder_name}/")
        return True

    def resumeAll(self, model_dict):
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.step = model_dict['step']
        self.eval_step = model_dict['eval_step']
        self.loss_min = model_dict['loss_min']
        self.eval_loss_min = model_dict['eval_loss_min']
        self.log_folder_name = model_dict['log_folder_name']
        return True

    def loadModel(self, model_file_path, resume_model_only=False):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        self.model.load_state_dict(model_dict['model'])

        if not resume_model_only:
            self.resumeAll(model_dict)

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_file_path):
        createFileFolder(save_file_path)

        appendix = '.' + save_file_path.split('.')[-1]
        base_file_path = save_file_path.split(appendix)[0]
        tmp_save_file_path = f'{base_file_path}_tmp{appendix}'

        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'eval_step': self.eval_step,
            'loss_min': self.loss_min,
            'eval_loss_min': self.eval_loss_min,
            'log_folder_name': self.log_folder_name,
        }

        torch.save(state_dict, tmp_save_file_path)

        removeFile(save_file_path)
        renameFile(tmp_save_file_path, save_file_path)
        return True

    def getLoss(self, batch_data, drop_one=0.1, drop_all=0.1, mode="train"):
        assert mode in ("train", "eval")

        # perform augmentation
        if mode == "train":
            batch_data = data_augmentation(batch_data)

        # get data
        rgbd = batch_data["rgbd"]
        pose = batch_data["pose"]
        intr = batch_data["intr"]

        # sample time steps
        t = torch.randint(
            0, self.diffusion_scheduler.config.num_train_timesteps,
            size=(len(rgbd), ), device=rgbd.device,
        )

        # make image noisy
        noise = torch.randn_like(rgbd[:, -1])
        # has been noised
        rgbd[:, -1] = self.diffusion_scheduler.add_noise(rgbd[:, -1], noise, t)

        # make it unconditional
        drop_msk = torch.rand(rgbd.shape[:2]) < drop_one  # drop some views
        drop_msk[torch.rand(len(rgbd)) < drop_all] = True  # drop all
        drop_msk[:, -1] = False  # don't drop current view
        rgbd[drop_msk] = 0.0  # drop by setting depth to zero

        # compute loss
        with torch.cuda.amp.autocast(enabled=self.fp16_mode):
            pred_noise = self.model(rgbd, (intr, pose), t)
            # compute loss
            loss_cor_dict = {"loss_color": F.l1_loss(
                pred_noise[:, :3], noise[:, :3])}
            loss_dep_dict = {"loss_depth": F.l1_loss(
                pred_noise[:, [3]], noise[:, [3]])}

        return loss_cor_dict | loss_dep_dict

    def trainStep(self, batch_data):
        self.model.train()

        # compute loss
        batch_data = move_to(batch_data, device=self.device)
        loss_dict = self.getLoss(batch_data, mode="train")
        loss_dict = combine_loss(loss_dict)

        # update model
        self.optimizer.zero_grad()
        if self.fp16_mode:
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict["loss"].backward()
            self.optimizer.step()

        data = dict(**{k: v.item() for k, v in loss_dict.items()},
                    lr=self.optimizer.param_groups[0]["lr"])
        print('trainStep', data)
        return True

    @ torch.no_grad()
    def evaluate_fn(self, val_loader):
        self.model.eval()

        count = 0.0
        total_loss_dict = defaultdict(float)
        # each process will compute
        for batch_data in tqdm(val_loader, desc="evaluating"):
            batch_size = len(batch_data["rgbd"])

            # compute loss
            batch_data = move_to(batch_data, device=self.device)
            loss_dict = self.eval_loss(batch_data, mode="eval")
            loss_dict = combine_loss(loss_dict)

            # update numbers
            count += batch_size
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() * batch_size  # sum

        # average
        return {k: v/count for k, v in total_loss_dict.items()}

    def evalStep(self):
        loss_dict = self.evaluate_fn(self.val_loader)
        print('evalStep', loss_dict)
        return True

    def train(self, print_progress=False):
        total_epoch = 10000000

        self.model.zero_grad()
        for epoch in range(total_epoch):
            self.summary_writer.add_scalar(
                "Lr/lr",
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                self.step)

            print("[INFO][Trainer::train]")
            print("\t start training, epoch : " + str(epoch + 1) + "/" +
                  str(total_epoch) + "...")
            for_data = self.train_loader
            if print_progress:
                for_data = tqdm(for_data)
            for data in for_data:
                self.trainStep(data)
                self.step += 1

            self.scheduler.step()

            print("[INFO][Trainer::train]")
            print("\t start evaling, epoch : " + str(epoch + 1) + "/" +
                  str(total_epoch) + "...")
            for_data = self.eval_dataloader
            if print_progress:
                for_data = tqdm(for_data)
            self.evalStep()
            self.eval_step += 1

            self.saveModel(f"./output/{self.log_folder_name}/model_last.pth")
        return True
