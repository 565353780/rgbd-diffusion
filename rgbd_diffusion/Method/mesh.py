import os
import os.path as osp

import cv2
import numpy as np
import open3d as o3d
import torch


def simplify_mesh(mesh, voxel_size):
    if voxel_size <= 0.0:
        return mesh
    device = mesh[0].device
    v, f, a = [item.cpu().numpy() for item in mesh]
    a = (a + 1.0) / 2.0  # to 0 ~ 1
    dtype_v = v.dtype
    dtype_f = f.dtype
    dtype_a = a.dtype
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(a.astype(np.float64))
    m = m.simplify_vertex_clustering(voxel_size=voxel_size)
    v = np.asarray(m.vertices).astype(dtype_v)
    f = np.asarray(m.triangles).astype(dtype_f)
    a = np.asarray(m.vertex_colors).astype(dtype_a)
    a = a * 2.0 - 1.0  # to -1 ~ +1
    v = torch.from_numpy(v).to(device=device)
    f = torch.from_numpy(f).to(device=device)
    a = torch.from_numpy(a).to(device=device)
    return v, f, a


def merge_mesh(mesh1, mesh2):
    v1, f1, a1 = mesh1
    v2, f2, a2 = mesh2
    v = torch.cat([v1, v2], dim=0)
    f = torch.cat([f1, f2 + len(v1)], dim=0)
    a = torch.cat([a1, a2], dim=0)
    return v, f, a


def empty_mesh(device):
    v = torch.zeros([0, 3], dtype=torch.float32, device=device)
    f = torch.zeros([0, 3], dtype=torch.long, device=device)
    a = torch.zeros([0, 3], dtype=torch.float32, device=device)
    return v, f, a


def save_mesh(path, mesh, model):
    if all(len(x) > 0 for x in mesh):
        model.save_mesh(path, mesh)
    else:
        WARN = "\033[91m[WARNING]\033[0m"  # the warning word
        print(f"{WARN} found mesh: {path} was empty, so we didn't save it.")


def save_files(save_folder, scene_name,
               rgbd_lst, mask_lst, cam_lst, known_lst,
               model, mean, std):
    os.makedirs(save_folder, exist_ok=True)
    all_mesh_lst = []
    for idx_view, (rgbd_i, mask_i, cam_i, known_i) in \
            enumerate(zip(rgbd_lst, mask_lst, cam_lst, known_lst)):
        rgbd_i = rgbd_i * std + mean
        #
        mesh_i = model.meshing(rgbd_i, *cam_i)
        all_mesh_lst.append(mesh_i)
        #
        # if "known", it means that it is a given view (don't need to generate)
        mark = "known" if known_i else "generation"
        mesh_i_name = f"{scene_name}_view{idx_view:03}_{mark}.ply"
        img_color_i_name = f"{scene_name}_view{idx_view:03}_color_{mark}.png"
        img_depth_i_name = f"{scene_name}_view{idx_view:03}_depth_{mark}.png"
        img_mask_i_name = f"{scene_name}_view{idx_view:03}_mask_{mark}.png"
        #
        save_mesh(osp.join(save_folder, mesh_i_name), mesh_i)
        #
        img_color_i_path = osp.join(save_folder, img_color_i_name)
        cv2.imwrite(img_color_i_path,
                    rgbd_i[..., [2, 1, 0]].add(1.0).mul(127.5).round()
                    .clamp(0, 255).cpu().numpy().astype(np.uint8),
                    )
        print(f"color image is saved to: {img_color_i_path}")
        #
        img_depth_i_path = osp.join(save_folder, img_depth_i_name)
        cv2.imwrite(img_depth_i_path,
                    rgbd_i[..., 3].mul(1000.0).round()
                    .clamp(0, 65535).cpu().numpy().astype(np.uint16),
                    )
        print(f"depth image is saved to: {img_depth_i_path}")
        #
        img_mask_i_path = osp.join(save_folder, img_mask_i_name)
        cv2.imwrite(img_mask_i_path, mask_i.byte().mul(255).cpu().numpy())
        print(f"mask image is saved to: {img_mask_i_path}")

    # save combined mesh
    mesh_combined = all_mesh_lst[0]
    for m in all_mesh_lst[1:]:
        mesh_combined = merge_mesh(mesh_combined, m)
        mesh_combined = simplify_mesh(mesh_combined)
    save_mesh(osp.join(save_folder, f"{scene_name}.ply"), mesh_combined)
