# NOTE
#   you can launch GUI by giving: "--interactive"
#     generate diverse results by varying camera trajectory/randomness
#     you can control the trajectory by button input

import math
import os
import os.path as osp
import sys

import torch
from einops import rearrange
from tqdm import trange

sys.path.append(".")

PIC_SIZE = 0.25
CTRL_WID = 0.25

BTN_HEI = 0.09
BTN_WID = 0.07
BTN_PAD_X = 0.01
BTN_PAD_Y = 0.04

DPI = 100
HEIGHT = 300
WIDTH = int(HEIGHT / PIC_SIZE)

INC_LOC = 0.15
INC_DEG = 10


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":
    sample_one_view = Sampler()

    if "%" in task:  # the percentage task
        perc = float(task.replace("%", "")) / 100.0  # 0~1
        assert 0 < perc <= 1

        # explore dataset
        num_scenes = len(dataset.info_chunk)
        ind_scenes_process = list()
        for ind_sce in range(num_scenes):
            if len(dataset.info_chunk[ind_sce]) >= min_views:
                ind_scenes_process.append(ind_sce)
        num_scenes_process = len(ind_scenes_process)
        print(f"there are {num_scenes} scenes in testset")
        print(
            f"but only {num_scenes_process} scenes have num_views >= {min_views}")
        print(f"their scene indices are: {ind_scenes_process}")

        # check if indices are out of bound
        if isinstance(ind_scenes, list):
            for ind in ind_scenes:
                assert 0 <= ind < num_scenes_process, \
                    f"only {num_scenes_process} scenes are available, " \
                    f"but you provide index = {ind}"

        # for each scene
        for ind_lst in (ind_scenes or range(num_scenes_process)):
            data = move_to(dataset[ind_scenes_process[ind_lst]], device=device)
            #
            num_views = len(data["rgbd"])
            num_views_down = math.ceil(num_views * perc)
            print(
                f"found {num_views} views in total, but we only use {num_views_down} views actually, due to percentage down-sampling.")
            ind_views_down = torch.linspace(
                0, num_views - 1, num_views_down).round().long().tolist()

            # first build the initial mesh using known views
            scene_mesh = empty_mesh()
            for ind_view in ind_views_down:  # for each view
                scene_mesh = merge_mesh(scene_mesh,
                                        model.meshing(
                                            rearrange(
                                                data["rgbd"][ind_view], "C H W -> H W C") * std + mean,
                                            data["intr"][ind_view], data["pose"][ind_view],
                                        )
                                        )
                scene_mesh = simplify_mesh(scene_mesh)

            # # # # # # # # # # # # # # # # # # # # # # # # # #
            # NOTE non-interactive
            if not interactive:
                # modify the camera trajectory (pose) here
                # example: "add_noise(std_loc=0.2,std_rot=0.1)|interpolate(between=2)"
                if traj != "":
                    commands = traj.split("|")
                    new_pose = data["pose"].clone()
                    ind_views_down = torch.tensor(ind_views_down)
                    ind_views_down_before = ind_views_down.clone()
                    for cmd in commands:
                        try:
                            new_pose, ind_insert = eval(
                                f"CameraTrajectoryTransform(new_pose, ind_views_down).{cmd}")
                            # get new index
                            ind_views_down = ind_insert[ind_views_down]
                        except Exception as e:
                            print(f"cannot parse command: {cmd}")
                            print(f"error message: {e}")
                            exit(0)
                    #
                    num_views = len(new_pose)
                    data["pose"] = new_pose
                    #
                    intr_original = data["intr"].clone()
                    data["intr"] = intr_original.mean(dim=0, keepdim=True).repeat(
                        [num_views, 1, 1])  # NOTE use the average intrinsic matrix
                    # visible views
                    data["intr"][ind_views_down] = intr_original[ind_views_down_before]
                    #
                    rgbd_original = data["rgbd"].clone()
                    data["rgbd"] = torch.zeros(
                        [num_views] + list(rgbd_original.shape[1:]), device=rgbd_original.device)
                    # visible views
                    data["rgbd"][ind_views_down] = rgbd_original[ind_views_down_before]
                    #
                    ind_views_down = ind_views_down.tolist()
                    num_views_down = len(ind_views_down)
                    #
                    print(
                        f"UPDATED TRAJECTORY: found {num_views} views in total, but we only use {num_views_down} views actually, due to percentage down-sampling.")

                # result storage
                rgbd_result = [None for _ in range(num_views)]
                mask_result = [None for _ in range(num_views)]

                # perform iteration
                for ind_view in trange(num_views, desc="moving camera"):
                    if ind_view in ind_views_down:  # already exists
                        rgbd_result[ind_view] = rearrange(
                            data["rgbd"][ind_view], "C H W -> H W C")
                        mask_result[ind_view] = torch.ones(
                            [IMG_SIZE, IMG_SIZE], dtype=torch.bool, device=device)
                    else:  # sampling for the novel view
                        cam_curr = (data["intr"][ind_view],
                                    data["pose"][ind_view])
                        rgbd_curr, mask_known = sample_one_view(
                            scene_mesh, cam_curr, seed=seed + ind_view,
                        )  # (H, W, 4)

                        # save current view
                        rgbd_result[ind_view] = rgbd_curr
                        mask_result[ind_view] = mask_known

                        # merge current mesh into `scene_mesh`
                        scene_mesh = merge_mesh(scene_mesh,
                                                model.meshing(
                                                    rgbd_curr * std + mean,
                                                    *cam_curr
                                                )
                                                )
                        scene_mesh = simplify_mesh(scene_mesh)

                # info
                cam_result = list(zip(data["intr"], data["pose"]))
                scene_name = f"scene{ind_scenes_process[ind_lst]:04}"

                # save pred
                save_files(osp.join(out_path, scene_name + suffix, "pred"), scene_name,
                           rgbd_result, mask_result, cam_result,
                           [(i in ind_views_down) for i in range(num_views)],
                           )

            # # # # # # # # # # # # # # # # # # # # # # # # # #
            else:  # NOTE interactive, you can control the trajectory
                scene_name = f"scene{ind_scenes_process[ind_lst]:04}"
                # prepare input views, and save as files
                rgbd_input_lst, mask_input_lst, cam_input_lst = [], [], []
                for ind_view in ind_views_down:
                    rgbd_input_lst += [rearrange(data["rgbd"]
                                                 [ind_view], "C H W -> H W C")]
                    mask_input_lst += [torch.ones([IMG_SIZE, IMG_SIZE],
                                                  dtype=torch.bool, device=device)]
                    cam_input_lst += [(data["intr"][ind_view],
                                       data["pose"][ind_view])]
                save_files(osp.join(out_path, scene_name + suffix, "pred", "view_input"), scene_name,
                           rgbd_input_lst, mask_input_lst, cam_input_lst,
                           [True for _ in ind_views_down],
                           )
                #
                cache_folder = osp.join(
                    out_path, scene_name + suffix, "pred", "interactive_output")
                os.makedirs(cache_folder, exist_ok=True)
                #

                def cam_to_rgbdm_fn(cam_curr):
                    global scene_mesh
                    rgb_out, d_out, vis_out = model.render_many(
                        (*scene_mesh,
                         torch.tensor([[0, len(scene_mesh[1]), 0, len(scene_mesh[0])]])),
                        (cam_curr[0][None, ...], cam_curr[1][None, ...]),
                        res=IMG_SIZE,
                    )
                    rgb_out = rearrange(rgb_out, "() C H W -> H W C")
                    rgb_out = (rgb_out + 1) / 2  # to 0~1
                    rgbdm = torch.cat([
                        rgb_out, d_out[0, ..., None], vis_out[0, ..., None],
                    ], dim=2)  # (H, W, 5)
                    return rgbdm
                # prepare initial image and camera
                init_rgbd = rearrange(
                    data["rgbd"][0], "C H W -> H W C") * std + mean
                init_rgbd[..., :3] = (init_rgbd[..., :3] + 1) / 2  # to 0~1
                init_rgbdm = torch.cat([
                    init_rgbd, torch.ones(
                        [IMG_SIZE, IMG_SIZE, 1], dtype=torch.bool, device=device)
                ], dim=2)
                init_cam = [data["intr"][0], data["pose"][0]]
                # buil GUI
                gui = Interface(
                    init_rgbdm, init_cam, cam_to_rgbdm_fn=cam_to_rgbdm_fn, cache_folder=cache_folder)
                gui.launch()  # pop out the window

            # save GT
            if save_gt:
                save_files(osp.join(out_path, scene_name + suffix, "gt"), scene_name,
                           [rearrange(rgbd, "C H W -> H W C")
                            for rgbd in data["rgbd"]],
                           [torch.ones([IMG_SIZE, IMG_SIZE],
                                       dtype=torch.bool, device=device)] * num_views,
                           cam_result, [True] * num_views,
                           )

            print("= = = = = = = = = = = = = = = = = = = = = = = =")

    else:
        raise
