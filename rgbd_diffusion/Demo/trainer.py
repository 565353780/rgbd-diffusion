from rgbd_diffusion.Module.trainer import Trainer


def demo():
    dataset_folder_path = '/home/chli/chLi/RGBD_Diffusion/scans_keyframe/'
    model_file_path = '/home/chli/chLi/RGBD_Diffusion/model.pt'

    trainer = Trainer(dataset_folder_path, model_file_path)
    return True
