# NOTE
#   you can launch GUI by giving: "--interactive"
#     generate diverse results by varying camera trajectory/randomness
#     you can control the trajectory by button input

import json
import math
import os
import os.path as osp
import sys
from collections import namedtuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation
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


class Interface:
    def __init__(self, rgbdm, cam, cam_to_rgbdm_fn, cache_folder=None):
        # save
        self.rgbdm = rgbdm  # NOTE (H, W, 5)   rgb:0~1   d:meters   m:0/1
        self.cam = cam
        self.cam_to_rgbdm_fn = cam_to_rgbdm_fn
        self.cache_folder = cache_folder  # output folder to save temporary results
        # how many times you press the inpaint button, it will change the randomness seed
        self.times_press_inpaint_btn = 0
        # index of fused view
        self.index_view = 0
        # backup, so that we can possibly recover the previous step
        self.back_up = None  # NOTE we only store one previous step
        # build GUI
        self.initialize_interface()

    # get each image prepared for display, output is RGB image in range 0~1
    def feed_display(self, what):
        assert what in ("rgb", "d", "m")
        rgbdm = self.rgbdm.clone()  # make sure you don't change it in place
        msk = rgbdm[..., 4] > 0.5
        if what == "rgb":
            mat = rgbdm[..., :3]
            mat[~msk] = 0
            mat = mat.detach().cpu().numpy()
        elif what == "d":
            mat = rgbdm[..., 3].clamp(min=0.1, max=10.0)
            mat[~msk] = 10.0
            mat = 1 / mat  # 0.1~10
            perc_3 = torch.quantile(mat.flatten(), 0.03)
            perc_97 = torch.quantile(mat.flatten(), 0.97)
            mat = mat.clamp(min=perc_3, max=perc_97)  # remove noise
            mat = (mat - mat.min()) / (mat.max() - mat.min())  # 0~1
            mapper = cm.ScalarMappable(cmap="magma")
            mat = mapper.to_rgba(mat.detach().cpu().numpy())[..., :3]
            mat[~msk.detach().cpu().numpy()] = 0
        else:
            mat = repeat(msk, "H W -> H W C", C=3).float()
            mat = mat.detach().cpu().numpy()
        return mat

    def initialize_interface(self):  # build the GUI interface
        #
        self.fig = plt.figure(num='RGBD Diffusion', figsize=(
            WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        self.fig.canvas.toolbar.pack_forget()  # remove status bar
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # color
        self.ax_color = self.fig.add_axes([0, 0, PIC_SIZE, 1])
        self.ax_color.set_axis_off()
        self.pic_color = self.ax_color.imshow(self.feed_display("rgb"))
        # depth
        self.ax_depth = self.fig.add_axes([PIC_SIZE, 0, PIC_SIZE, 1])
        self.ax_depth.set_axis_off()
        self.pic_depth = self.ax_depth.imshow(self.feed_display("d"))
        # binary mask
        self.ax_mask = self.fig.add_axes([PIC_SIZE * 2, 0, PIC_SIZE, 1])
        self.ax_mask.set_axis_off()
        self.pic_mask = self.ax_mask.imshow(self.feed_display("m"))
        #
        # NOTE immediately launch rendering
        self.update_pose_and_render(self.cam[1])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # translation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 0.94,
                      "Translation Control:", fontsize=10)
        #
        self.ax_move_left = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_left = Button(self.ax_move_left, 'Left')
        self.btn_move_left.on_clicked(self.move_left)
        # #
        self.ax_move_back = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_back = Button(self.ax_move_back, 'Back')
        self.btn_move_back.on_clicked(self.move_back)
        # #
        self.ax_move_right = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_right = Button(self.ax_move_right, 'Right')
        self.btn_move_right.on_clicked(self.move_right)
        # #
        self.ax_move_forw = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_forw = Button(self.ax_move_forw, 'Forward')
        self.btn_move_forw.on_clicked(self.move_forw)
        # #
        self.ax_move_down = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_down = Button(self.ax_move_down, 'Down')
        self.btn_move_down.on_clicked(self.move_down)
        # #
        self.ax_move_up = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_up = Button(self.ax_move_up, 'Up')
        self.btn_move_up.on_clicked(self.move_up)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # rotation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 1.0 -
                      (BTN_HEI + BTN_PAD_Y) * 3, "Rotation Control:", fontsize=10)
        #
        self.ax_rota_left = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_left = Button(self.ax_rota_left, 'Left')
        self.btn_rota_left.on_clicked(self.rota_left)
        # #
        self.ax_rota_down = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_down = Button(self.ax_rota_down, 'Down')
        self.btn_rota_down.on_clicked(self.rota_down)
        # #
        self.ax_rota_right = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_right = Button(self.ax_rota_right, 'Right')
        self.btn_rota_right.on_clicked(self.rota_right)
        # #
        self.ax_rota_up = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_up = Button(self.ax_rota_up, 'Up')
        self.btn_rota_up.on_clicked(self.rota_up)
        #
        self.ax_rota_ccw = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_ccw = Button(self.ax_rota_ccw, 'CCW')
        self.btn_rota_ccw.on_clicked(self.rota_ccw)
        # #
        self.ax_rota_cw = self.fig.add_axes(
            [1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_cw = Button(self.ax_rota_cw, 'CW')
        self.btn_rota_cw.on_clicked(self.rota_cw)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # rotation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 1.05 -
                      (BTN_HEI + BTN_PAD_Y) * 6, "Operation:", fontsize=10)
        #
        self.ax_inpaint = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (
            BTN_HEI + BTN_PAD_Y) * 7 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_inpaint = Button(self.ax_inpaint, 'Inpaint')
        self.btn_inpaint.on_clicked(self.op_inpaint)
        #
        self.ax_done = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1.5, 1 - (
            BTN_HEI + BTN_PAD_Y) * 7 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_done = Button(self.ax_done, 'All Done')
        self.btn_done.on_clicked(self.op_done)
        #
        self.ax_prev = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (
            BTN_HEI + BTN_PAD_Y) * 8 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_prev = Button(self.ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.op_prev)
        #
        self.ax_next = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1.5, 1 - (
            BTN_HEI + BTN_PAD_Y) * 8 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.op_next)
        #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def decompose_pose(self, camext):
        right = camext[:, 0]
        up = camext[:, 1].neg()
        lookat = camext[:, 2]
        loc = camext[:, 3]
        return namedtuple("CameraCoordSystem", ["right", "up", "lookat", "loc"])(right, up, lookat, loc)

    def display_rgbdm(self):  # display to window
        # display color image to GUI
        self.pic_color.set_data(self.feed_display("rgb"))
        # display depth image to GUI
        self.pic_depth.set_data(self.feed_display("d"))
        # display mask image to GUI
        self.pic_mask.set_data(self.feed_display("m"))

    def update_pose_and_render(self, pose_new):
        self.cam = (self.cam[0], pose_new)  # update pose
        self.rgbdm = self.cam_to_rgbdm_fn(self.cam)  # render image
        self.display_rgbdm()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def move_left(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)

    def move_right(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)

    def move_back(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)

    def move_forw(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)

    def move_down(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)

    def move_up(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def rota_left(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("y", -INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_right(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("y", INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_down(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("x", -INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_up(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("x", INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_ccw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("z", INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_cw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) *
                        Rotation.from_euler("z", -INC_DEG, degrees=True)
                        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def op_inpaint(self, _):
        # NOTE only inpaint and visualize,
        # but do not supplement to the mesh,
        # and do not save to file;
        # you can press the button for many times to get desirable results
        #
        rgbd_curr, mask_known = sample_one_view(
            scene_mesh, self.cam,
            seed=self.times_press_inpaint_btn,
        )  # (H, W, 4)
        # NOTE we inpaint all the pixels, so the mask should be ideally all ones
        mask_known.fill_(1.0)
        #
        rgbd_curr = rgbd_curr * std + mean
        rgbd_curr[..., :3] = (rgbd_curr[..., :3] + 1) / 2  # 0~1
        rgbd_curr[..., :3] = rgbd_curr[..., :3].clamp(min=0, max=1)
        rgbd_curr[...,  3] = rgbd_curr[...,  3].clamp(min=0)
        #
        self.rgbdm = torch.cat([
            rgbd_curr, mask_known[..., None].float()
        ], dim=2)  # (H, W, 5)
        #
        self.times_press_inpaint_btn += 1
        self.display_rgbdm()  # show

    def op_next(self, _):
        # NOTE supplement to the mesh,
        # also save to file
        #
        global scene_mesh
        self.back_up = dict(  # all are in CPU to save memory
            index_view=self.index_view,
            times_press_inpaint_btn=self.times_press_inpaint_btn,
            scene_mesh=[t.cpu().numpy() for t in scene_mesh],
            cam=[t.cpu().numpy() for t in self.cam],
            rgbdm=self.rgbdm.cpu().numpy(),
        )
        #
        rgbdm = self.rgbdm.clone()  # retrieve the just rendered image
        rgbdm[..., :3] = 2 * rgbdm[..., :3] - 1  # -1 ~ 1
        rgbd = rgbdm[..., :4]
        # merge current mesh into `scene_mesh`
        scene_mesh = merge_mesh(scene_mesh,
                                model.meshing(rgbd, *self.cam)
                                )
        scene_mesh = simplify_mesh(scene_mesh)
        # refresh the image immediately
        self.update_pose_and_render(self.cam[1])
        # save to cache folder
        if self.cache_folder is not None:
            # save mesh
            save_mesh(osp.join(self.cache_folder,
                      f"mesh_{self.index_view}.ply"), scene_mesh)
            # save color
            img_color_path = osp.join(
                self.cache_folder, f"color_{self.index_view}.png")
            cv2.imwrite(img_color_path,
                        rgbd[..., [2, 1, 0]].add(1.0).mul(127.5).round()
                        .clamp(0, 255).cpu().numpy().astype(np.uint8),
                        )
            print(f"color image is saved to: {img_color_path}")
            # save depth
            img_depth_path = osp.join(
                self.cache_folder, f"depth_{self.index_view}.png")
            cv2.imwrite(img_depth_path,
                        rgbd[..., 3].mul(1000.0).round()
                        .clamp(0, 65535).cpu().numpy().astype(np.uint16),
                        )
            print(f"depth image is saved to: {img_depth_path}")
            # save camera pose as a json file
            cam_path = osp.join(self.cache_folder,
                                f"camera_{self.index_view}.json")
            with open(cam_path, "w") as f:
                json.dump(dict(
                    camint=self.cam[0].cpu().numpy().tolist(),
                    camext=self.cam[1].cpu().numpy().tolist(),
                ), f, indent=4, sort_keys=True)
            print(f"camera parameter is saved to: {cam_path}")
        # next view
        self.index_view += 1

    def op_prev(self, _):  # equivalent to "ctrl+z", go back to the previous result
        if self.back_up is not None:
            global scene_mesh
            # recover others
            self.index_view = self.back_up["index_view"]
            self.times_press_inpaint_btn = self.back_up["times_press_inpaint_btn"]
            scene_mesh = [torch.from_numpy(t).to(
                device=device) for t in self.back_up["scene_mesh"]]
            self.cam = [torch.from_numpy(t).to(device=device)
                        for t in self.back_up["cam"]]
            self.rgbdm = torch.from_numpy(
                self.back_up["rgbdm"]).to(device=device)
            # we also remove those saved files
            for p in [osp.join(self.cache_folder, f"mesh_{self.index_view}.ply"),
                      osp.join(self.cache_folder,
                               f"color_{self.index_view}.png"),
                      osp.join(self.cache_folder,
                               f"depth_{self.index_view}.png"),
                      osp.join(self.cache_folder,
                               f"camera_{self.index_view}.json"),
                      ]:
                if osp.exists(p):
                    os.remove(p)
            # clear backup
            self.back_up = None
            # render immediately
            self.update_pose_and_render(self.cam[1])
            print("successfully restore a previous step")
        else:
            WARN = "\033[91m[WARNING]\033[0m"  # the warning word
            print(
                f"{WARN} cannot restore the previous step, because there is no backup.")

    def op_done(self, _):
        plt.close()  # close the window

    def launch(self):
        plt.show()  # pop out the window


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
