#!/bin/bash
set -e

pip install numpy==1.23.5
pip install pandas==1.4.2
pip install Pillow==11.3.0
pip install imageio==2.37.2
pip install imgaug==0.4.0
pip install tqdm==4.61.0
pip install scipy==1.13.1
pip install seaborn==0.11.2
pip install matplotlib==3.9.4
pip install pyyaml==6.0
pip install imutils==0.5.4
pip install scikit-image==0.19.2
pip install scikit-learn==1.0.2
pip install albumentations==1.1.0
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install efficientnet-pytorch==0.7.1
pip install timm==0.6.12
pip install segmentation-models-pytorch==0.3.2
pip install torchtoolbox==0.1.8.2
pip install tensorboard==2.10.1
pip install setuptools==59.5.0
pip install loralib==0.1.2
pip install einops==0.8.2
pip install transformers==4.30.2 tokenizers==0.13.3
pip install filterpy==1.4.5
pip install simplejson==4.2.0
pip install kornia==0.6.9
pip install fvcore==0.1.5.post20221221
pip install git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless==4.6.0.66
