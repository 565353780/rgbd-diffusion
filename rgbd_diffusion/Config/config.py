import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_loss(batch_data, drop_one=0.1, drop_all=0.1, mode="train"):
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
        0, diffusion_scheduler.config.num_train_timesteps,
        size=(len(rgbd), ), device=rgbd.device,
    )

    # make image noisy
    noise = torch.randn_like(rgbd[:, -1])
    # has been noised
    rgbd[:, -1] = diffusion_scheduler.add_noise(rgbd[:, -1], noise, t)

    # make it unconditional
    drop_msk = torch.rand(rgbd.shape[:2]) < drop_one  # drop some views
    drop_msk[torch.rand(len(rgbd)) < drop_all] = True  # drop all
    drop_msk[:, -1] = False  # don't drop current view
    rgbd[drop_msk] = 0.0  # drop by setting depth to zero

    # compute loss
    with torch.cuda.amp.autocast(enabled=FP16_MODE):
        pred_noise = model(rgbd, (intr, pose), t)
        # compute loss
        loss_cor_dict = {"loss_color": F.l1_loss(
            pred_noise[:, :3], noise[:, :3])}
        loss_dep_dict = {"loss_depth": F.l1_loss(
            pred_noise[:, [3]], noise[:, [3]])}

    return {**loss_cor_dict, **loss_dep_dict}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def move_to(d, device):
    d_out = dict()
    for k, v in d.items():
        d_out[k] = v.to(device=device) if isinstance(v, torch.Tensor) else v
    return d_out


def combine_loss(loss_dict):
    loss_dict["loss"] = 0.20 * loss_dict["loss_color"] + \
        0.80 * loss_dict["loss_depth"]  # depth is very important
    return loss_dict


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def train_step_fn(batch_data):
    model.train()

    # compute loss
    batch_data = move_to(batch_data, device=device)
    loss_dict = eval_loss(batch_data, mode="train")
    loss_dict = combine_loss(loss_dict)

    # update model
    optimizer.zero_grad()
    if FP16_MODE:
        scaler.scale(loss_dict["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict["loss"].backward()
        optimizer.step()

    return dict(**{k: v.item() for k, v in loss_dict.items()},
                lr=optimizer.param_groups[0]["lr"])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@torch.no_grad()
def evaluate_fn(val_loader):  # NOTE evaluating the losses on test set
    model.eval()

    count = 0.0
    total_loss_dict = defaultdict(float)
    for batch_data in tqdm(val_loader, desc="evaluating"):  # each process will compute
        batch_size = len(batch_data["rgbd"])

        # compute loss
        batch_data = move_to(batch_data, device=device)
        loss_dict = eval_loss(batch_data, mode="eval")
        loss_dict = combine_loss(loss_dict)

        # update numbers
        count += batch_size
        for k, v in loss_dict.items():
            total_loss_dict[k] += v.item() * batch_size  # sum

    # average
    return {k: v/count for k, v in total_loss_dict.items()}
