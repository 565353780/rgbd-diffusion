import nvdiffrast.torch as dr
import torch
import trimesh
from einops import rearrange


class Render(object):
    def __init__(self, extend_eps=0.04, depth_min=0.1, edge_max=0.1):
        super().__init__()
        self.extend_eps = extend_eps
        self.depth_min = depth_min  # in meters
        self.edge_max = edge_max  # in meters
        self.ctx_render = dr.RasterizeCudaContext()  # require CUDA

    @staticmethod
    def gmm(grouped_left, right, ind_group):
        """ perform grouped matrix multiplication
        """
        result = []
        for ind, (ind_start, cnt) in enumerate(ind_group):
            mat_left = grouped_left[ind_start: (ind_start + cnt)]
            mat_right = right[ind]
            mat_out = mat_left @ mat_right
            result.append(mat_out)
        return torch.cat(result, dim=0)

    def unproject(self, attr_d, c2w_int, c2w_ext, flatten=True):
        """ unproject a 2d image into 3d space
        Args:
            attr_d.shape == (H, W, num_attr + 1) or (H, W)
            c2w_int.shape == (3, 3)
            c2w_ext.shape == (3, 4)
            flatten: whether to flatten (H, W) to (H*W, )
        """
        device = attr_d.device
        H, W = attr_d.shape[:2]
        if attr_d.ndim == 2:
            attr_d = attr_d[..., None]
        num_attr = attr_d.size(2) - 1

        def make_linspace(num):
            lin = torch.linspace(0.5, num - 0.5, num, device=device)
            lin[-1] += self.extend_eps  # extend upper boundary location
            return lin
        x, y = torch.meshgrid(make_linspace(W),
                              make_linspace(H),
                              indexing="xy",
                              )  # each == (H, W)
        z = attr_d[..., -1]  # (H, W)
        a = None
        if num_attr > 0:
            a = attr_d[..., :-1]  # (H, W, num_attr)
            a[-1, :-1] += (a[-1, :-1] - a[-2, :-1]) * self.extend_eps
            a[:-1,  -1] += (a[:-1,  -1] - a[:-1,  -2]) * self.extend_eps
            a[-1,  -1] += (a[-1,  -1] - a[-2,  -2]) * self.extend_eps
        #
        v = torch.stack([x * z, y * z, z], dim=2)  # (H, W, 3)
        v = torch.einsum("ij,...j->...i", c2w_int, v)  # (H, W, 3)
        v = torch.cat([v, v.new_ones([H, W, 1])], dim=2)  # (H, W, 4)
        v = torch.einsum("ij,...j->...i", c2w_ext, v)  # (H, W, 3)
        if flatten:
            v = rearrange(v, "H W C -> (H W) C")
            if num_attr > 0:
                a = rearrange(a, "H W C -> (H W) C")
        return v, a

    @staticmethod
    def mesh_structure(H, W, device="cpu"):
        x, y = torch.meshgrid(torch.arange(W, device=device),
                              torch.arange(H, device=device),
                              indexing="xy",
                              )  # each == (H, W)
        pts = y[:-1, :-1].mul(W) + x[:-1, :-1]
        lower = torch.stack([pts, pts + W, pts + W + 1], dim=2).reshape(-1, 3)
        upper = torch.stack([pts, pts + W + 1, pts + 1], dim=2).reshape(-1, 3)
        return torch.cat([lower, upper], dim=0)

    def meshing(self, attr_d, c2w_int, c2w_ext, trunc=True):
        """ meshing a 2d image into a 3d mesh
        Args:
            attr_d.shape == (H, W, num_attr + 1) or (H, W)
            c2w_int.shape == (3, 3)
            c2w_ext.shape == (3, 4)
        """
        v, a = self.unproject(attr_d, c2w_int, c2w_ext, flatten=True)
        f = self.mesh_structure(*attr_d.shape[:2], device=attr_d.device)
        if trunc:
            # remove those with small depth values
            d = (attr_d if attr_d.ndim ==
                 2 else attr_d[..., -1]).reshape(-1)  # (H*W, )
            msk0 = (d[f] > self.depth_min).all(dim=1)  # (num_faces, )
            # remove those with too long edges
            edge_len = torch.stack([
                (v[f[:, 0]] - v[f[:, 1]]).norm(dim=1),
                (v[f[:, 1]] - v[f[:, 2]]).norm(dim=1),
                (v[f[:, 2]] - v[f[:, 0]]).norm(dim=1),
            ], dim=1)  # (num_faces, 3)
            msk1 = (edge_len < self.edge_max).all(dim=1)  # (num_faces, )
            # remove faces
            f = f[msk0 & msk1]
            # remove verts
            ind_v, f = f.reshape(-1).unique(return_inverse=True)
            v = v[ind_v]
            a = a[ind_v]
            f = f.reshape(-1, 3)
        return v, f, a  # vertices, faces, attributes

    def meshing_many(self, attr_d, c2w_int, c2w_ext, trunc=True):
        """
        meshing many images into many meshes,
        stacking into a batch ready for rendering
        Args:
            attr_d.shape == (B, num_attr + 1, H, W)
            c2w_int.shape == (B, 3, 3)
            c2w_ext.shape == (B, 3, 4)
        Returns:
            mesh (tuple): containing (vertices, faces, attrs, ind_group)
        """
        has_attr = attr_d.size(1) > 1
        vertices, faces, attrs = [], [], []
        offset = 0
        for attrd, camint, camext in zip(attr_d, c2w_int, c2w_ext):
            attrd = rearrange(attrd, "C H W -> H W C")  # (H, W, num_attr + 1)
            v, f, a = self.meshing(attrd, camint, camext, trunc=trunc)
            vertices.append(v)
            faces.append(f + offset)
            if has_attr:
                attrs.append(a)
            offset += len(v)
        num_faces = torch.tensor([len(f) for f in faces])  # on CPU
        num_verts = torch.tensor([len(v) for v in vertices])  # on CPU
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces,    dim=0)
        attrs = torch.cat(attrs,    dim=0) if has_attr else None
        ind_group = torch.stack([
            # for faces
            # start index
            torch.cat([torch.tensor([0]), num_faces.cumsum(dim=0)[:-1]]),
            num_faces,  # count
            # for vertices
            # start index
            torch.cat([torch.tensor([0]), num_verts.cumsum(dim=0)[:-1]]),
            num_verts,  # count
        ], dim=1)  # (batch_size, 4), on CPU
        return vertices, faces, attrs, ind_group

    @staticmethod
    def save_mesh(path, mesh):
        """ save one mesh
        """
        v, f, a = mesh
        assert a.shape[1] == 3, "attributes should have 3 channels (RGB)"
        if len(v) == 0 or len(f) == 0:  # no mesh found
            print(f"mesh is empty, cannot save to: {path}")
            return
        if a.max() < 5:  # guess it should be converted into 0~255
            a = (a + 1.0) * 127.5
        mesh = trimesh.Trimesh(
            vertices=v.cpu().numpy(),
            faces=f.cpu().numpy(),
            vertex_colors=a.clamp(0, 255).byte().cpu().numpy(),
        )
        mesh.export(path)
        print(f"mesh is saved to: {path}")

    @staticmethod
    def save_meshes(path, meshes):
        """ save many meshes
        """
        assert "{i" in path  # e.g. path == "mesh_{i:05}.ply"
        vv, ff, aa, group = meshes
        for ind_mesh, (f_st, f_cnt, v_st, v_cnt) in enumerate(group):
            v = vv[v_st:(v_st + v_cnt)]
            f = ff[f_st:(f_st + f_cnt)] - v_st
            a = aa[v_st:(v_st + v_cnt)]
            Render.save_mesh(path.format(i=ind_mesh), (v, f, a))

    @staticmethod
    def c2w_to_w2c(c2w_int, c2w_ext):
        """
        Args:
            c2w_int.shape == (3, 3) or (..., 3, 3)
            c2w_ext.shape == (3, 4) or (..., 3, 4)
        """
        w2c_int = torch.linalg.inv(c2w_int)  # (..., 3, 3)
        rot_inv = rearrange(c2w_ext[..., :3], "... I J -> ... J I")
        w2c_ext = torch.cat([
            rot_inv,
            rot_inv @ c2w_ext[..., [3]].neg(),
        ], dim=-1)  # (..., 3, 4)
        return w2c_int, w2c_ext

    def render_many(self, pack_meshes, pack_cameras, res=128):
        """ render many meshes
        Returns:
            img_attr.shape == (B, 3+1, H, W), float32
            img_dep.shape == (B, H, W), float32
            img_vib.shape == (B, H, W), bool
        """
        vertices, faces, attrs, ind_group = pack_meshes
        w2c_int, w2c_ext = self.c2w_to_w2c(*pack_cameras)

        # for each mesh
        f_idx, v_idx = 0, 0
        z_mins, z_maxs = [], []
        f_new, v_new, a_new = [], [], []
        # TODO it may be slow, but rather straightforward
        for ind_mesh, (f_st, f_cnt, v_st, v_cnt) in enumerate(ind_group):
            f = faces[f_st:(f_st + f_cnt)] - v_st
            v = vertices[v_st:(v_st + v_cnt)]
            a = attrs[v_st:(v_st + v_cnt)]

            # project vertices
            v = torch.cat([v, v.new_ones([len(v), 1])], dim=1)
            v = torch.einsum("ik,jk->ij", v, w2c_ext[ind_mesh])
            v = torch.einsum("ik,jk->ij", v, w2c_int[ind_mesh])
            #
            z = v[:,  2]
            uv = v[:, :2] / z[:, None]
            uv = (uv - res/2) / (res/2)  # to -1 ~ +1
            #
            msk_v = (z > self.depth_min) & (uv.abs() < 1).all(
                dim=1)  # must be inside the bbox
            # TODO: msk_v may all False
            assert True in msk_v, 'msk_v is all False!'

            # TODO maybe `any` is better,
            # but compute of `msk_v` is a bit tricky
            msk_f = msk_v[f].all(dim=1)
            #
            num_v = msk_v.sum().item()
            num_f = msk_f.sum().item()
            #
            if msk_v.any():
                z_min = z[msk_v].min().item()
                z_max = z[msk_v].max().item()
            else:  # when the mesh is empty
                z_min, z_max = 0.0, 1.0
            z_mins.append(z_min)
            z_maxs.append(z_max)
            #
            w = (z[msk_v] - z_min) / (z_max - z_min) * 2.0 - 1.0  # to -1 ~ +1
            v_new.append(
                torch.cat([uv[msk_v], w[:, None]], dim=1)
            )
            #
            a_new.append(
                a[msk_v]
            )
            #
            pi = msk_v.nonzero().squeeze(1)
            p = torch.zeros_like(msk_v, dtype=torch.long)
            p[pi] = torch.arange(len(pi), device=p.device)
            f_new.append(
                p[f[msk_f]] + v_idx
            )
            #
            ind_group[ind_mesh, 0] = f_idx
            ind_group[ind_mesh, 1] = num_f
            ind_group[ind_mesh, 2] = v_idx
            ind_group[ind_mesh, 3] = num_v
            #
            f_idx += num_f
            v_idx += num_v

        # combine results
        coord_clip = torch.cat(v_new, dim=0)
        coord_clip = torch.cat(
            [coord_clip, coord_clip.new_ones([len(coord_clip), 1])], dim=1)
        faces = torch.cat(f_new, dim=0)
        attrs = torch.cat(a_new, dim=0)
        del v_new, f_new, a_new  # it may take lots of memory, so delete them

        # depth's range
        z_max = torch.tensor(z_maxs, device=vertices.device)[:, None, None]
        z_min = torch.tensor(z_mins, device=vertices.device)[:, None, None]

        # render
        # has already on CPU
        rast, _ = dr.rasterize(self.ctx_render,
                               coord_clip.float().contiguous(),
                               faces.to(torch.int32).contiguous(),
                               resolution=(res, res),
                               ranges=ind_group[:, :2].to(
                                   torch.int32).contiguous(),
                               )
        # attribute
        img_attr, _ = dr.interpolate(
            attrs.contiguous(),
            rast,
            faces.to(torch.int32).contiguous(),
        )
        img_attr = rearrange(img_attr, "B H W C -> B C H W")
        # depth
        img_dep = (rast[..., 2] + 1.0) / 2.0  # to 0~1
        img_dep = img_dep * (z_max - z_min) + z_min  # (B, H, W)
        # visibility
        img_vib = rast.any(dim=3)  # bool, (B, H, W)
        return img_attr, img_dep, img_vib

    def unproject_render_many(self, c2w_curr, rgbd_prev, c2w_prev,
                              res=128, trunc=True):
        """ project all the previous views onto the current view
        Args:
            c2w_curr.shape == (B, 3, 3) and (B, 3, 4)
            rgbd_prev.shape == (B, C, H, W)
            c2w_prev.shape == (B, 3, 3) and (B, 3, 4)
        Returns:
            rgbd_out.shape == (B, C, H, W)
            mask_out.shape == (B, H, W)
        """
        meshes_prev = self.meshing_many(rgbd_prev, *c2w_prev, trunc=trunc)
        img_attr, img_dep, mask_out = self.render_many(
            meshes_prev, c2w_curr, res=res,
        )
        # merge
        rgbd_out = torch.cat([img_attr, rearrange(
            img_dep, "B H W -> B () H W")], dim=1)  # (B, C=4, H, W)
        return rgbd_out, mask_out
