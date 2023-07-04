pip install matplotlib opencv-python einops \
	trimesh diffusers ninja open3d tensorboard

pip install torch torchvision torchaudio

pip install git+https://github.com/wookayin/gpustat.git@master

cd ./third_party/latent-diffusion/
pip install .

cd ../nvdiffrast/
pip install .

cd ../recon/
pip install .
