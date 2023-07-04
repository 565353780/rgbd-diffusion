import json
import os
import os.path as osp
from collections import namedtuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat
from matplotlib.widgets import Button
from rgbd_diffusion.Method.mesh import merge_mesh, save_mesh, simplify_mesh
from scipy.spatial.transform import Rotation


class Interface:
    def __init__(self, rgbdm, cam, cam_to_rgbdm_fn, cache_folder=None):
        self.pic_size = 0.25
        self.ctrl_wid = 0.25

        self.btn_hei = 0.09
        self.btn_wid = 0.07
        self.btn_pad_x = 0.01
        self.btn_pad_y = 0.04

        self.dpi = 100
        self.height = 300
        self.width = int(self.height / self.pic_size)

        self.inc_loc = 0.15
        self.inc_deg = 10

        # save
        self.rgbdm = rgbdm  # NOTE (H, W, 5)   rgb:0~1   d:meters   m:0/1
        self.cam = cam
        self.cam_to_rgbdm_fn = cam_to_rgbdm_fn
        self.cache_folder = cache_folder
        # how many times press inpaint button, change the randomness seed
        self.times_press_inpaint_btn = 0
        # index of fused view
        self.index_view = 0
        # backup, so that we can possibly recover the previous step
        self.back_up = None  # NOTE we only store one previous step
        # build GUI
        self.initialize_interface()
        return

    def extracteDepth(self, rgbdm, msk):
        result = rgbdm[..., 3].clamp(min=0.1, max=10.0)
        result[~msk] = 10.0
        result = 1 / result
        perc_3 = torch.quantile(result.flatten(), 0.03)
        perc_97 = torch.quantile(result.flatten(), 0.97)
        result = result.clamp(min=perc_3, max=perc_97)
        result = (result - result.min()) / (result.max() - result.min())
        mapper = cm.ScalarMappable(cmap="magma")
        result = mapper.to_rgba(result.detach().cpu().numpy())[..., :3]
        result[~msk.detach().cpu().numpy()] = 0
        return result

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
            mat = self.extracteDepth(rgbdm, msk)
        else:
            mat = repeat(msk, "H W -> H W C", C=3).float()
            mat = mat.detach().cpu().numpy()
        return mat

    def initialize_interface(self):  # build the GUI interface
        #
        self.fig = plt.figure(num='RGBD Diffusion', figsize=(
            self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.fig.canvas.toolbar.pack_forget()  # remove status bar
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # color
        self.ax_color = self.fig.add_axes([0, 0, self.pic_size, 1])
        self.ax_color.set_axis_off()
        self.pic_color = self.ax_color.imshow(self.feed_display("rgb"))
        # depth
        self.ax_depth = self.fig.add_axes([self.pic_size, 0, self.pic_size, 1])
        self.ax_depth.set_axis_off()
        self.pic_depth = self.ax_depth.imshow(self.feed_display("d"))
        # binary mask
        self.ax_mask = self.fig.add_axes(
            [self.pic_size * 2, 0, self.pic_size, 1])
        self.ax_mask.set_axis_off()
        self.pic_mask = self.ax_mask.imshow(self.feed_display("m"))
        #
        # NOTE immediately launch rendering
        self.update_pose_and_render(self.cam[1])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # translation control
        self.fig.text(1 - (self.btn_wid + self.btn_pad_x) * 3, 0.94,
                      "Translation Control:", fontsize=10)
        #
        self.ax_move_left = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 3,
             1 - (self.btn_hei + self.btn_pad_y) * 2 - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_left = Button(self.ax_move_left, 'Left')
        self.btn_move_left.on_clicked(self.move_left)
        # #
        self.ax_move_back = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 3,
             1 - (self.btn_hei + self.btn_pad_y) - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_back = Button(self.ax_move_back, 'Back')
        self.btn_move_back.on_clicked(self.move_back)
        # #
        self.ax_move_right = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 1,
             1 - (self.btn_hei + self.btn_pad_y) * 2 - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_right = Button(self.ax_move_right, 'Right')
        self.btn_move_right.on_clicked(self.move_right)
        # #
        self.ax_move_forw = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 1,
             1 - (self.btn_hei + self.btn_pad_y) - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_forw = Button(self.ax_move_forw, 'Forward')
        self.btn_move_forw.on_clicked(self.move_forw)
        # #
        self.ax_move_down = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 2,
             1 - (self.btn_hei + self.btn_pad_y) * 2 - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_down = Button(self.ax_move_down, 'Down')
        self.btn_move_down.on_clicked(self.move_down)
        # #
        self.ax_move_up = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 2,
             1 - (self.btn_hei + self.btn_pad_y) - 0.05,
             self.btn_wid, self.btn_hei])
        self.btn_move_up = Button(self.ax_move_up, 'Up')
        self.btn_move_up.on_clicked(self.move_up)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # rotation control
        self.fig.text(1 - (self.btn_wid + self.btn_pad_x) * 3,
                      1.0 - (self.btn_hei + self.btn_pad_y) * 3,
                      "Rotation Control:", fontsize=10)
        #
        self.ax_rota_left = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 3,
             1 - (self.btn_hei + self.btn_pad_y) * 5 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_left = Button(self.ax_rota_left, 'Left')
        self.btn_rota_left.on_clicked(self.rota_left)
        # #
        self.ax_rota_down = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 2,
             1 - (self.btn_hei + self.btn_pad_y) * 5 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_down = Button(self.ax_rota_down, 'Down')
        self.btn_rota_down.on_clicked(self.rota_down)
        # #
        self.ax_rota_right = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 1,
             1 - (self.btn_hei + self.btn_pad_y) * 5 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_right = Button(self.ax_rota_right, 'Right')
        self.btn_rota_right.on_clicked(self.rota_right)
        # #
        self.ax_rota_up = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 2,
             1 - (self.btn_hei + self.btn_pad_y) * 4 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_up = Button(self.ax_rota_up, 'Up')
        self.btn_rota_up.on_clicked(self.rota_up)
        #
        self.ax_rota_ccw = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 3,
             1 - (self.btn_hei + self.btn_pad_y) * 4 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_ccw = Button(self.ax_rota_ccw, 'CCW')
        self.btn_rota_ccw.on_clicked(self.rota_ccw)
        # #
        self.ax_rota_cw = self.fig.add_axes(
            [1 - (self.btn_wid + self.btn_pad_x) * 1,
             1 - (self.btn_hei + self.btn_pad_y) * 4 + 0.01,
             self.btn_wid, self.btn_hei])
        self.btn_rota_cw = Button(self.ax_rota_cw, 'CW')
        self.btn_rota_cw.on_clicked(self.rota_cw)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # rotation control
        self.fig.text(1 - (self.btn_wid + self.btn_pad_x) * 3,
                      1.05 - (self.btn_hei + self.btn_pad_y) * 6,
                      "Operation:", fontsize=10)
        #
        self.ax_inpaint = self.fig.add_axes([
            1 - (self.btn_wid + self.btn_pad_x) * 3,
            1 - (self.btn_hei + self.btn_pad_y) * 7 + 0.07,
            1.5 * self.btn_wid + 0.5 * self.btn_pad_x, self.btn_hei])
        self.btn_inpaint = Button(self.ax_inpaint, 'Inpaint')
        self.btn_inpaint.on_clicked(self.op_inpaint)
        #
        self.ax_done = self.fig.add_axes([
            1 - (self.btn_wid + self.btn_pad_x) * 1.5,
            1 - (self.btn_hei + self.btn_pad_y) * 7 + 0.07,
            1.5 * self.btn_wid + 0.5 * self.btn_pad_x, self.btn_hei])
        self.btn_done = Button(self.ax_done, 'All Done')
        self.btn_done.on_clicked(self.op_done)
        #
        self.ax_prev = self.fig.add_axes([
            1 - (self.btn_wid + self.btn_pad_x) * 3,
            1 - (self.btn_hei + self.btn_pad_y) * 8 + 0.07,
            1.5 * self.btn_wid + 0.5 * self.btn_pad_x, self.btn_hei])
        self.btn_prev = Button(self.ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.op_prev)
        #
        self.ax_next = self.fig.add_axes([
            1 - (self.btn_wid + self.btn_pad_x) * 1.5,
            1 - (self.btn_hei + self.btn_pad_y) * 8 + 0.07,
            1.5 * self.btn_wid + 0.5 * self.btn_pad_x, self.btn_hei])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.op_next)
        #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def decompose_pose(self, camext):
        right = camext[:, 0]
        up = camext[:, 1].neg()
        lookat = camext[:, 2]
        loc = camext[:, 3]
        return namedtuple("CameraCoordSystem",
                          ["right", "up", "lookat", "loc"])(
            right, up, lookat, loc)

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
        camext_new[:, 3] -= self.inc_loc * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)

    def move_right(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += self.inc_loc * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)

    def move_back(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= self.inc_loc * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)

    def move_forw(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += self.inc_loc * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)

    def move_down(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= self.inc_loc * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)

    def move_up(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += self.inc_loc * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def rota_left(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("y", -self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_right(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("y", self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_down(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("x", -self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_up(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("x", self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_ccw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("z", self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_cw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(
            camext[:, :3].detach().cpu().numpy()) *
            Rotation.from_euler("z", -self.inc_deg, degrees=True)
        ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def op_inpaint(self, sample_func, mean, std, _):
        # NOTE only inpaint and visualize,
        # but do not supplement to the mesh,
        # and do not save to file;
        # you can press the button for many times to get desirable results
        #
        rgbd_curr, mask_known = sample_func(
            scene_mesh, self.cam,
            seed=self.times_press_inpaint_btn,
        )  # (H, W, 4)
        # NOTE we inpaint all the pixels,
        # so the mask should be ideally all ones
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

    def saveCache(self, scene_mesh, model, rgbd):
        # save mesh
        save_mesh(osp.join(self.cache_folder,
                  f"mesh_{self.index_view}.ply"), scene_mesh, model)
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
        return True

    def op_next(self, voxel_size, model, _):
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
        scene_mesh = simplify_mesh(scene_mesh, voxel_size)
        # refresh the image immediately
        self.update_pose_and_render(self.cam[1])
        # save to cache folder
        if self.cache_folder is not None:
            self.saveCache(scene_mesh, model, rgbd)
        # next view
        self.index_view += 1

    # equivalent to "ctrl+z", go back to the previous result
    def op_prev(self, device, _):
        if self.back_up is not None:
            global scene_mesh
            # recover others
            self.index_view = self.back_up["index_view"]
            self.times_press_inpaint_btn = self.back_up[
                "times_press_inpaint_btn"]
            scene_mesh = [torch.from_numpy(t).to(
                device=device) for t in self.back_up["scene_mesh"]]
            self.cam = [torch.from_numpy(t).to(device=device)
                        for t in self.back_up["cam"]]
            self.rgbdm = torch.from_numpy(
                self.back_up["rgbdm"]).to(device=device)
            # we also remove those saved files
            for p in [osp.join(self.cache_folder,
                               f"mesh_{self.index_view}.ply"),
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
                f"{WARN} cannot restore the previous step, \
                because there is no backup.")

    def op_done(self, _):
        plt.close()  # close the window

    def launch(self):
        plt.show()  # pop out the window
