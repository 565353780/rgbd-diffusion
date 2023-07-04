import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


class CameraTrajectoryTransform:
    # NOTE must be adjacent in time sequence
    def __init__(self, camext_lst, ind_dont_change):
        assert isinstance(camext_lst, torch.Tensor)  # (B, 3, 4)
        assert isinstance(ind_dont_change, torch.Tensor)
        self.kwargs = dict(dtype=camext_lst.dtype, device=camext_lst.device)
        self.camext_lst = camext_lst.detach().cpu().numpy()
        self.ind_dont_change = ind_dont_change.detach().cpu().numpy()

    def interpolate(self, between=5):
        assert between >= 0
        num = len(self.camext_lst)
        #
        t_in = np.arange(num)
        t_out = np.linspace(0, num - 1, num + (num - 1) * between)
        # interpolate rotations
        new_R = Slerp(t_in, Rotation.from_matrix(self.camext_lst[:, :, :3]))(
            t_out).as_matrix()  # (num, 3, 3)
        # interpolate translations
        new_T = interp1d(t_in, self.camext_lst[:, :, 3].T)(t_out).T  # (num, 3)
        #
        new_RT = np.concatenate([new_R, new_T[:, :, None]], axis=2)
        ind_insert = torch.from_numpy(t_in * (between + 1)).long()  # CPU
        return torch.from_numpy(new_RT).to(**self.kwargs), ind_insert

    def add_noise(self, std_loc=0.2, std_rot=0.1):
        # add noise to rotations
        quat = Rotation.from_matrix(self.camext_lst[:, :, :3]).as_quat()
        quat += np.random.randn(*quat.shape) * std_rot
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        new_R = Rotation.from_quat(quat).as_matrix()
        # add translations
        new_T = self.camext_lst[:, :, 3].copy()
        new_T += np.random.randn(*new_T.shape) * std_loc
        #
        new_RT = np.concatenate([new_R, new_T[:, :, None]], axis=2)
        # don't change them
        new_RT[self.ind_dont_change] = self.camext_lst[self.ind_dont_change]
        ind_insert = torch.arange(len(self.camext_lst)).long()  # CPU
        return torch.from_numpy(new_RT).to(**self.kwargs), ind_insert
