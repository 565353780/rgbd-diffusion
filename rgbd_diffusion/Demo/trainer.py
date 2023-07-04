from rgbd_diffusion.Module.trainer import Trainer


def demo():
    dataset_folder_path = '/home/chli/chLi/RGBD_Diffusion/scans_keyframe/'
    model_file_path = '/home/chli/chLi/RGBD_Diffusion/model.pt'
    resume_model_only = True
    print_progress = True

    trainer = Trainer(dataset_folder_path, model_file_path, resume_model_only)
    trainer.train(print_progress)
    return True
