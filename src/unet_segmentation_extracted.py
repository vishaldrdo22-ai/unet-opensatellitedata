
Requirement already satisfied: patchify in ./.local/lib/python3.10/site-packages (0.2.3)
Requirement already satisfied: numpy<2,>=1 in ./.local/lib/python3.10/site-packages (from patchify) (1.26.4)

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: wandb in ./.local/lib/python3.10/site-packages (0.22.1)
Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (3.1.45)
Requirement already satisfied: packaging in ./.local/lib/python3.10/site-packages (from wandb) (25.0)
Requirement already satisfied: pydantic<3 in ./.local/lib/python3.10/site-packages (from wandb) (2.11.9)
Requirement already satisfied: pyyaml in /usr/lib/python3/dist-packages (from wandb) (5.4.1)
Requirement already satisfied: sentry-sdk>=2.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (2.39.0)
Requirement already satisfied: requests<3,>=2.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (2.32.5)
Requirement already satisfied: typing-extensions<5,>=4.8 in ./.local/lib/python3.10/site-packages (from wandb) (4.15.0)
Requirement already satisfied: platformdirs in ./.local/lib/python3.10/site-packages (from wandb) (4.4.0)
Requirement already satisfied: click>=8.0.1 in /usr/lib/python3/dist-packages (from wandb) (8.0.3)
Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in ./.local/lib/python3.10/site-packages (from wandb) (4.25.8)
Requirement already satisfied: gitdb<5,>=4.0.1 in ./.local/lib/python3.10/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb) (4.0.12)
Requirement already satisfied: pydantic-core==2.33.2 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (2.33.2)
Requirement already satisfied: annotated-types>=0.6.0 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (0.7.0)
Requirement already satisfied: typing-inspection>=0.4.0 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (0.4.1)
Requirement already satisfied: charset_normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (3.4.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests<3,>=2.0.0->wandb) (2020.6.20)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests<3,>=2.0.0->wandb) (3.3)
Requirement already satisfied: smmap<6,>=3.0.1 in ./.local/lib/python3.10/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb) (5.0.2)

# ===== Cell Separator =====

Name: wandb
Version: 0.22.1
Summary: A CLI and library for interacting with the Weights & Biases API.
Home-page:
Author:
Author-email: Weights & Biases <support@wandb.com>
License: MIT License

        Copyright (c) 2021 Weights and Biases, Inc.

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
Location: /home/user/.local/lib/python3.10/site-packages
Requires: click, gitpython, packaging, platformdirs, protobuf, pydantic, pyyaml, requests, sentry-sdk, typing-extensions
Required-by:
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

import os
import wandb

os.environ["WANDB_API_KEY"] = "7245e2a7430898865fcd5a6404a16858f83cc588"
wandb.login()
#!wandb login --relogin

# ===== Cell Separator =====

import wandb
from wandb.integration.keras import WandbCallback

# ===== Cell Separator =====

2025-10-06 10:29:30.514845: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-10-06 10:29:30.843401: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-10-06 10:29:30.843444: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-10-06 10:29:30.900248: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-10-06 10:29:31.008794: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-06 10:29:31.864627: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

# ===== Cell Separator =====

import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: opencv-python in ./.local/lib/python3.10/site-packages (4.12.0.88)
Collecting numpy<2.3.0,>=2
  Using cached numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
Installing collected packages: numpy
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.4
    Uninstalling numpy-1.26.4:
      Successfully uninstalled numpy-1.26.4
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.15.1 requires numpy<2.0.0,>=1.23.5, but you have numpy 2.2.6 which is incompatible.
patchify 0.2.3 requires numpy<2,>=1, but you have numpy 2.2.6 which is incompatible.
gradio 3.41.2 requires numpy~=1.0, but you have numpy 2.2.6 which is incompatible.
Successfully installed numpy-2.2.6
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
  dir_name = path.split(os.path.sep)[-1]
  #print(dir_name)
  if dir_name == 'masks': # 'images
    images = os.listdir(path)
    print(path)
    #print(images)
    for i, image_name in enumerate(images):
      if (image_name.endswith('.png')): # '.jpg
        #print(image_name)
        a = True

# ===== Cell Separator =====

/home/user/Downloads/archive/Semanticdataset/Tile 1/masks
/home/user/Downloads/archive/Semanticdataset/Tile 3/masks
/home/user/Downloads/archive/Semanticdataset/Tile 2/masks
/home/user/Downloads/archive/Semanticdataset/Tile 4/masks
/home/user/Downloads/archive/Semanticdataset/Tile 5/masks
/home/user/Downloads/archive/Semanticdataset/Tile 6/masks
/home/user/Downloads/archive/Semanticdataset/Tile 7/masks
/home/user/Downloads/archive/Semanticdataset/Tile 8/masks

# ===== Cell Separator =====

minmaxscaler = MinMaxScaler()
image_x = image_patches[0,0,:,:]
#MinMaxScaler
image_y = minmaxscaler.fit_transform(image_x.reshape(-1, image_x.shape[-1])).reshape(image_x.shape)
image_y[0].shape
print(type(image))
type(Image.fromarray(image))
image.shape
(image.shape[0]//image_patch_size)*image_patch_size

# ===== Cell Separator =====

image_dataset = []
mask_dataset = []

for image_type in ['images' , 'masks']:
  if image_type == 'images':
    image_extension = 'jpg'
  elif image_type == 'masks':
     image_extension = 'png'
  for tile_id in range(1,8):
    for image_id in range(1,20):
      image = cv2.imread(f'{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}',1)
      if image is not None:
        if image_type == 'masks':
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        size_x = (image.shape[1]//image_patch_size)*image_patch_size
        size_y = (image.shape[0]//image_patch_size)*image_patch_size
        #print("{} --- {} - {}".format(image.shape, size_x, size_y))
        image = Image.fromarray(image)
        image = image.crop((0,0, size_x, size_y))
        #print("({},  {})".format(image.size[0],image.size[1]))
        image = np.array(image)
        patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
        #print(len(patched_images))
        for i in range(patched_images.shape[0]):
          for j in range(patched_images.shape[1]):
            if image_type == 'images':
              individual_patched_image = patched_images[i,j,:,:]
              #print(individual_patched_image.shape)
              individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
              individual_patched_image = individual_patched_image[0]
              #print(individual_patched_image.shape)
              image_dataset.append(individual_patched_image)
            elif image_type == 'masks':
              individual_patched_mask = patched_images[i,j,:,:]
              individual_patched_mask = individual_patched_mask[0]
              mask_dataset.append(individual_patched_mask)

# ===== Cell Separator =====

[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.695] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.772] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@3.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.352] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@4.918] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@5.236] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0010.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0011.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0012.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0013.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0014.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0015.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0016.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0017.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0018.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.234] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/images/image_part_0019.jpg'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.271] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 1/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.280] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 2/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.304] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 3/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.339] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 4/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.406] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 5/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.443] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 6/masks/image_part_0019.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0010.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0011.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0012.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0013.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0014.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0015.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0016.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0017.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0018.png'): can't open/read file: check file path/integrity
[ WARN:0@7.751] global loadsave.cpp:275 findDecoder imread_('/home/user/Downloads/archive//Semanticdataset/Tile 7/masks/image_part_0019.png'): can't open/read file: check file path/integrity

# ===== Cell Separator =====

print(len(image_dataset))
print(len(mask_dataset))

# ===== Cell Separator =====

945
945

# ===== Cell Separator =====

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# ===== Cell Separator =====

print(len(image_dataset))
print(len(mask_dataset))

# ===== Cell Separator =====

945
945

# ===== Cell Separator =====

random_image_id = np.random.randint(0, len(image_dataset))

plt.figure(figsize=(14,10))
plt.subplot(1,2,1)
plt.imshow(image_dataset[random_image_id])
plt.subplot(1,2,2)
plt.imshow(mask_dataset[random_image_id])

# ===== Cell Separator =====

class_building = '#3C1098'
class_building = class_building.lstrip('#')
class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))
print(class_building)

class_land = '#8429F6'
class_land = class_land.lstrip('#')
class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))
print(class_land)

class_road = '#6EC1E4'
class_road = class_road.lstrip('#')
class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))
print(class_road)

class_vegetation = '#FEDD3A'
class_vegetation = class_vegetation.lstrip('#')
class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))
print(class_vegetation)

class_water = '#E2A929'
class_water = class_water.lstrip('#')
class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))
print(class_water)

class_unlabeled = '#9B9B9B'
class_unlabeled = class_unlabeled.lstrip('#')
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))
print(class_unlabeled)

# ===== Cell Separator =====

[ 60  16 152]
[132  41 246]
[110 193 228]
[254 221  58]
[226 169  41]
[155 155 155]

# ===== Cell Separator =====

def rgb_to_label(label):
  label_segment = np.zeros(label.shape, dtype=np.uint8)
  label_segment[np.all(label == class_water, axis=-1)] = 0
  label_segment[np.all(label == class_land, axis=-1)] = 1
  label_segment[np.all(label == class_road, axis=-1)] = 2
  label_segment[np.all(label == class_building, axis=-1)] = 3
  label_segment[np.all(label == class_vegetation, axis=-1)] = 4
  label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
  #print(label_segment)
  label_segment = label_segment[:,:,0]
  #print(label_segment)
  return label_segment

# ===== Cell Separator =====

labels = []
for i in range(mask_dataset.shape[0]):
  label = rgb_to_label(mask_dataset[i])
  labels.append(label)

# ===== Cell Separator =====

array([[1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       ...,
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1]], dtype=uint8)

# ===== Cell Separator =====

array([[[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]],

       [[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]],

       [[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]],

       ...,

       [[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]],

       [[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]],

       [[1],
        [1],
        [1],
        ...,
        [1],
        [1],
        [1]]], dtype=uint8)

# ===== Cell Separator =====

random_image_id = np.random.randint(0, len(image_dataset))

plt.figure(figsize=(14,10))
plt.subplot(1,2,1)
plt.imshow(image_dataset[random_image_id])
plt.subplot(1,2,2)
#plt.imshow(mask_dataset[random_image_id])
plt.imshow(labels[random_image_id])

# ===== Cell Separator =====

array([[1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       ...,
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1],
       [1, 1, 1, ..., 1, 1, 1]], dtype=uint8)

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow in ./.local/lib/python3.10/site-packages (2.15.1)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (2.15.0)
Requirement already satisfied: packaging in ./.local/lib/python3.10/site-packages (from tensorflow) (25.0)
Requirement already satisfied: ml-dtypes~=0.3.1 in ./.local/lib/python3.10/site-packages (from tensorflow) (0.3.2)
Requirement already satisfied: six>=1.12.0 in /usr/lib/python3/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in ./.local/lib/python3.10/site-packages (from tensorflow) (0.37.1)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in ./.local/lib/python3.10/site-packages (from tensorflow) (1.75.0)
Requirement already satisfied: absl-py>=1.0.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (2.3.1)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (1.14.2)
Requirement already satisfied: tensorboard<2.16,>=2.15 in ./.local/lib/python3.10/site-packages (from tensorflow) (2.15.2)
Requirement already satisfied: libclang>=13.0.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (18.1.1)
Requirement already satisfied: h5py>=2.9.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (3.14.0)
Requirement already satisfied: opt-einsum>=2.3.2 in ./.local/lib/python3.10/site-packages (from tensorflow) (3.4.0)
Requirement already satisfied: termcolor>=1.1.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (3.1.0)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in ./.local/lib/python3.10/site-packages (from tensorflow) (0.6.0)
Requirement already satisfied: keras<2.16,>=2.15.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (2.15.0)
Requirement already satisfied: astunparse>=1.6.0 in ./.local/lib/python3.10/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: setuptools in /usr/lib/python3/dist-packages (from tensorflow) (59.6.0)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in ./.local/lib/python3.10/site-packages (from tensorflow) (4.25.8)
Requirement already satisfied: flatbuffers>=23.5.26 in ./.local/lib/python3.10/site-packages (from tensorflow) (25.2.10)
Requirement already satisfied: typing-extensions>=3.6.6 in ./.local/lib/python3.10/site-packages (from tensorflow) (4.15.0)
Requirement already satisfied: google-pasta>=0.1.1 in ./.local/lib/python3.10/site-packages (from tensorflow) (0.2.0)
Collecting numpy<2.0.0,>=1.23.5
  Using cached numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/lib/python3/dist-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
Requirement already satisfied: werkzeug>=1.0.1 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.1.3)
Requirement already satisfied: requests<3,>=2.21.0 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.32.5)
Requirement already satisfied: google-auth<3,>=1.6.3 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (2.41.1)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (1.2.2)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (0.7.2)
Requirement already satisfied: markdown>=2.6.8 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow) (3.9)
Requirement already satisfied: cachetools<7.0,>=2.0.0 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (6.2.0)
Requirement already satisfied: rsa<5,>=3.1.4 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (4.9.1)
Requirement already satisfied: pyasn1-modules>=0.2.1 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.4.2)
Requirement already satisfied: requests-oauthlib>=0.7.0 in ./.local/lib/python3.10/site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (2.0.0)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (2020.6.20)
Requirement already satisfied: charset_normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.4.3)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow) (3.3)
Requirement already satisfied: MarkupSafe>=2.1.1 in ./.local/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow) (2.1.5)
Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in ./.local/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow) (0.6.1)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/lib/python3/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow) (3.2.0)
Installing collected packages: numpy
  Attempting uninstall: numpy
    Found existing installation: numpy 2.2.6
    Uninstalling numpy-2.2.6:
      Successfully uninstalled numpy-2.2.6
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
Successfully installed numpy-1.26.4
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# ===== Cell Separator =====

(803, 256, 256, 3)
(142, 256, 256, 3)
(803, 256, 256, 6)
(142, 256, 256, 6)

# ===== Cell Separator =====

image_height = X_train.shape[1]
image_width = X_train.shape[2]
image_channels = X_train.shape[3]
total_classes = y_train.shape[3]

# ===== Cell Separator =====

print(image_height)
print(image_width)
print(image_channels)
print(total_classes)

# ===== Cell Separator =====

256
256
3
6

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: segmentation-models in ./.local/lib/python3.10/site-packages (1.0.1)
Requirement already satisfied: keras-applications<=1.0.8,>=1.0.7 in ./.local/lib/python3.10/site-packages (from segmentation-models) (1.0.8)
Requirement already satisfied: image-classifiers==1.0.0 in ./.local/lib/python3.10/site-packages (from segmentation-models) (1.0.0)
Requirement already satisfied: efficientnet==1.0.0 in ./.local/lib/python3.10/site-packages (from segmentation-models) (1.0.0)
Requirement already satisfied: scikit-image in ./.local/lib/python3.10/site-packages (from efficientnet==1.0.0->segmentation-models) (0.25.2)
Requirement already satisfied: h5py in ./.local/lib/python3.10/site-packages (from keras-applications<=1.0.8,>=1.0.7->segmentation-models) (3.14.0)
Requirement already satisfied: numpy>=1.9.1 in ./.local/lib/python3.10/site-packages (from keras-applications<=1.0.8,>=1.0.7->segmentation-models) (1.26.4)
Requirement already satisfied: pillow>=10.1 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (10.4.0)
Requirement already satisfied: imageio!=2.35.0,>=2.33 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (2.37.0)
Requirement already satisfied: tifffile>=2022.8.12 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (2025.5.10)
Requirement already satisfied: lazy-loader>=0.4 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (0.4)
Requirement already satisfied: networkx>=3.0 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (3.4.2)
Requirement already satisfied: scipy>=1.11.4 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (1.15.3)
Requirement already satisfied: packaging>=21 in ./.local/lib/python3.10/site-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (25.0)

# ===== Cell Separator =====

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda

# ===== Cell Separator =====

def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

# ===== Cell Separator =====

def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):

  inputs = Input((image_height, image_width, image_channels))

  source_input = inputs

  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
  c1 = Dropout(0.2)(c1)
  c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
  p1 = MaxPooling2D((2,2))(c1)

  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
  c2 = Dropout(0.2)(c2)
  c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
  p2 = MaxPooling2D((2,2))(c2)

  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
  p3 = MaxPooling2D((2,2))(c3)

  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
  p4 = MaxPooling2D((2,2))(c4)

  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
  c5 = Dropout(0.2)(c5)
  c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

  u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

  u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

  u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
  c8 = Dropout(0.2)(c8)
  c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

  u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
  c9 = Dropout(0.2)(c9)
  c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

  outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model

# ===== Cell Separator =====

print(image_height)
print(image_width)
print(image_channels)
print(total_classes)

# ===== Cell Separator =====

256
256
3
6

# ===== Cell Separator =====

def get_deep_learning_model():
  return multi_unet_model(n_classes=total_classes,
                          image_height=image_height,
                          image_width=image_width,
                          image_channels=image_channels)

# ===== Cell Separator =====

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # makes TF expose Keras 2.x API
os.environ["SM_FRAMEWORK"] = "tf.keras"  # tell segmentation_models to use tf.keras

# ===== Cell Separator =====

dice_loss = sm.losses.DiceLoss(class_weights = weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# ===== Cell Separator =====

import tensorflow as tf
tf.keras.backend.clear_session()
model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
model.summary()

# ===== Cell Separator =====

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]        0         []

 conv2d (Conv2D)             (None, 256, 256, 16)         448       ['input_1[0][0]']

 dropout (Dropout)           (None, 256, 256, 16)         0         ['conv2d[0][0]']

 conv2d_1 (Conv2D)           (None, 256, 256, 16)         2320      ['dropout[0][0]']

 max_pooling2d (MaxPooling2  (None, 128, 128, 16)         0         ['conv2d_1[0][0]']
 D)

 conv2d_2 (Conv2D)           (None, 128, 128, 32)         4640      ['max_pooling2d[0][0]']

 dropout_1 (Dropout)         (None, 128, 128, 32)         0         ['conv2d_2[0][0]']

 conv2d_3 (Conv2D)           (None, 128, 128, 32)         9248      ['dropout_1[0][0]']

 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 32)           0         ['conv2d_3[0][0]']
 g2D)

 conv2d_4 (Conv2D)           (None, 64, 64, 64)           18496     ['max_pooling2d_1[0][0]']

 dropout_2 (Dropout)         (None, 64, 64, 64)           0         ['conv2d_4[0][0]']

 conv2d_5 (Conv2D)           (None, 64, 64, 64)           36928     ['dropout_2[0][0]']

 max_pooling2d_2 (MaxPoolin  (None, 32, 32, 64)           0         ['conv2d_5[0][0]']
 g2D)

 conv2d_6 (Conv2D)           (None, 32, 32, 128)          73856     ['max_pooling2d_2[0][0]']

 dropout_3 (Dropout)         (None, 32, 32, 128)          0         ['conv2d_6[0][0]']

 conv2d_7 (Conv2D)           (None, 32, 32, 128)          147584    ['dropout_3[0][0]']

 max_pooling2d_3 (MaxPoolin  (None, 16, 16, 128)          0         ['conv2d_7[0][0]']
 g2D)

 conv2d_8 (Conv2D)           (None, 16, 16, 256)          295168    ['max_pooling2d_3[0][0]']

 dropout_4 (Dropout)         (None, 16, 16, 256)          0         ['conv2d_8[0][0]']

 conv2d_9 (Conv2D)           (None, 16, 16, 256)          590080    ['dropout_4[0][0]']

 conv2d_transpose (Conv2DTr  (None, 32, 32, 128)          131200    ['conv2d_9[0][0]']
 anspose)

 concatenate (Concatenate)   (None, 32, 32, 256)          0         ['conv2d_transpose[0][0]',
                                                                     'conv2d_7[0][0]']

 conv2d_10 (Conv2D)          (None, 32, 32, 128)          295040    ['concatenate[0][0]']

 dropout_5 (Dropout)         (None, 32, 32, 128)          0         ['conv2d_10[0][0]']

 conv2d_11 (Conv2D)          (None, 32, 32, 128)          147584    ['dropout_5[0][0]']

 conv2d_transpose_1 (Conv2D  (None, 64, 64, 64)           32832     ['conv2d_11[0][0]']
 Transpose)

 concatenate_1 (Concatenate  (None, 64, 64, 128)          0         ['conv2d_transpose_1[0][0]',
 )                                                                   'conv2d_5[0][0]']

 conv2d_12 (Conv2D)          (None, 64, 64, 64)           73792     ['concatenate_1[0][0]']

 dropout_6 (Dropout)         (None, 64, 64, 64)           0         ['conv2d_12[0][0]']

 conv2d_13 (Conv2D)          (None, 64, 64, 64)           36928     ['dropout_6[0][0]']

 conv2d_transpose_2 (Conv2D  (None, 128, 128, 32)         8224      ['conv2d_13[0][0]']
 Transpose)

 concatenate_2 (Concatenate  (None, 128, 128, 64)         0         ['conv2d_transpose_2[0][0]',
 )                                                                   'conv2d_3[0][0]']

 conv2d_14 (Conv2D)          (None, 128, 128, 32)         18464     ['concatenate_2[0][0]']

 dropout_7 (Dropout)         (None, 128, 128, 32)         0         ['conv2d_14[0][0]']

 conv2d_15 (Conv2D)          (None, 128, 128, 32)         9248      ['dropout_7[0][0]']

 conv2d_transpose_3 (Conv2D  (None, 256, 256, 16)         2064      ['conv2d_15[0][0]']
 Transpose)

 concatenate_3 (Concatenate  (None, 256, 256, 32)         0         ['conv2d_transpose_3[0][0]',
 )                                                                   'conv2d_1[0][0]']

 conv2d_16 (Conv2D)          (None, 256, 256, 16)         4624      ['concatenate_3[0][0]']

 dropout_8 (Dropout)         (None, 256, 256, 16)         0         ['conv2d_16[0][0]']

 conv2d_17 (Conv2D)          (None, 256, 256, 16)         2320      ['dropout_8[0][0]']

 conv2d_18 (Conv2D)          (None, 256, 256, 6)          102       ['conv2d_17[0][0]']

==================================================================================================
Total params: 1941190 (7.41 MB)
Trainable params: 1941190 (7.41 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: graphviz in ./.local/lib/python3.10/site-packages (0.21)
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

import keras
from IPython.display import clear_output

%matplotlib inline

# ===== Cell Separator =====

class PlotLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait= True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


plot_loss = PlotLoss()

# ===== Cell Separator =====

import wandb
from wandb.integration.keras import WandbCallback

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: wandb in ./.local/lib/python3.10/site-packages (0.22.1)
Requirement already satisfied: pyyaml in /usr/lib/python3/dist-packages (from wandb) (5.4.1)
Requirement already satisfied: requests<3,>=2.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (2.32.5)
Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (3.1.45)
Requirement already satisfied: sentry-sdk>=2.0.0 in ./.local/lib/python3.10/site-packages (from wandb) (2.39.0)
Requirement already satisfied: typing-extensions<5,>=4.8 in ./.local/lib/python3.10/site-packages (from wandb) (4.15.0)
Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in ./.local/lib/python3.10/site-packages (from wandb) (4.25.8)
Requirement already satisfied: platformdirs in ./.local/lib/python3.10/site-packages (from wandb) (4.4.0)
Requirement already satisfied: pydantic<3 in ./.local/lib/python3.10/site-packages (from wandb) (2.11.9)
Requirement already satisfied: packaging in ./.local/lib/python3.10/site-packages (from wandb) (25.0)
Requirement already satisfied: click>=8.0.1 in /usr/lib/python3/dist-packages (from wandb) (8.0.3)
Requirement already satisfied: gitdb<5,>=4.0.1 in ./.local/lib/python3.10/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb) (4.0.12)
Requirement already satisfied: annotated-types>=0.6.0 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (0.7.0)
Requirement already satisfied: typing-inspection>=0.4.0 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (0.4.1)
Requirement already satisfied: pydantic-core==2.33.2 in ./.local/lib/python3.10/site-packages (from pydantic<3->wandb) (2.33.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2.5.0)
Requirement already satisfied: charset_normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (3.4.3)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests<3,>=2.0.0->wandb) (3.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests<3,>=2.0.0->wandb) (2020.6.20)
Requirement already satisfied: smmap<6,>=3.0.1 in ./.local/lib/python3.10/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb) (5.0.2)
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: keras<3 in ./.local/lib/python3.10/site-packages (2.15.0)
Requirement already satisfied: tensorflow<2.16 in ./.local/lib/python3.10/site-packages (2.15.1)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (1.26.4)
Requirement already satisfied: typing-extensions>=3.6.6 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (4.15.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (0.37.1)
Requirement already satisfied: packaging in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (25.0)
Requirement already satisfied: libclang>=13.0.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (18.1.1)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (0.6.0)
Requirement already satisfied: h5py>=2.9.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (3.14.0)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (1.14.2)
Requirement already satisfied: astunparse>=1.6.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (1.6.3)
Requirement already satisfied: six>=1.12.0 in /usr/lib/python3/dist-packages (from tensorflow<2.16) (1.16.0)
Requirement already satisfied: absl-py>=1.0.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (2.3.1)
Requirement already satisfied: ml-dtypes~=0.3.1 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (0.3.2)
Requirement already satisfied: flatbuffers>=23.5.26 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (25.2.10)
Requirement already satisfied: google-pasta>=0.1.1 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (0.2.0)
Requirement already satisfied: setuptools in /usr/lib/python3/dist-packages (from tensorflow<2.16) (59.6.0)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (2.15.0)
Requirement already satisfied: opt-einsum>=2.3.2 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (3.4.0)
Requirement already satisfied: tensorboard<2.16,>=2.15 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (2.15.2)
Requirement already satisfied: termcolor>=1.1.0 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (3.1.0)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (4.25.8)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in ./.local/lib/python3.10/site-packages (from tensorflow<2.16) (1.75.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/lib/python3/dist-packages (from astunparse>=1.6.0->tensorflow<2.16) (0.37.1)
Requirement already satisfied: google-auth<3,>=1.6.3 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (2.41.1)
Requirement already satisfied: werkzeug>=1.0.1 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (3.1.3)
Requirement already satisfied: requests<3,>=2.21.0 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (2.32.5)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (0.7.2)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (1.2.2)
Requirement already satisfied: markdown>=2.6.8 in ./.local/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow<2.16) (3.9)
Requirement already satisfied: rsa<5,>=3.1.4 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16) (4.9.1)
Requirement already satisfied: pyasn1-modules>=0.2.1 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16) (0.4.2)
Requirement already satisfied: cachetools<7.0,>=2.0.0 in ./.local/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16) (6.2.0)
Requirement already satisfied: requests-oauthlib>=0.7.0 in ./.local/lib/python3.10/site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<2.16) (2.0.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16) (3.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16) (2020.6.20)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16) (2.5.0)
Requirement already satisfied: charset_normalizer<4,>=2 in ./.local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<2.16) (3.4.3)
Requirement already satisfied: MarkupSafe>=2.1.1 in ./.local/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow<2.16) (2.1.5)
Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in ./.local/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<2.16) (0.6.1)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/lib/python3/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<2.16) (3.2.0)
Note: you may need to restart the kernel to use updated packages.

# ===== Cell Separator =====

model_history = model.fit(X_train, y_train,
                          batch_size=16,
                          verbose=1,
                          epochs=250,
                          validation_data=(X_test, y_test),
                          callbacks = [WandbCallback(save_graph=False)],
                          shuffle=False)
#model.save("my_model.keras")

# ===== Cell Separator =====

wandb: WARNING WandbCallback is deprecated and will be removed in a future release. Please use the WandbMetricsLogger, WandbModelCheckpoint, and WandbEvalCallback callbacks instead. See https://docs.wandb.ai/guides/integrations/keras for more information.
wandb: WARNING The save_model argument by default saves the model in the HDF5 format that cannot save custom objects like subclassed models and custom layers. This behavior will be deprecated in a future release in favor of the SavedModel format. Meanwhile, the HDF5 model is saved as W&B files and the SavedModel as W&B Artifacts.

# ===== Cell Separator =====

Epoch 1/250
51/51 [==============================] - ETA: 0s - loss: 1.0054 - accuracy: 0.5147 - jaccard_coef: 0.2418

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 74s 1s/step - loss: 1.0054 - accuracy: 0.5147 - jaccard_coef: 0.2418 - val_loss: 0.9909 - val_accuracy: 0.5881 - val_jaccard_coef: 0.3375
Epoch 2/250
51/51 [==============================] - ETA: 0s - loss: 0.9797 - accuracy: 0.6650 - jaccard_coef: 0.3812

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 70s 1s/step - loss: 0.9797 - accuracy: 0.6650 - jaccard_coef: 0.3812 - val_loss: 0.9694 - val_accuracy: 0.7014 - val_jaccard_coef: 0.4542
Epoch 3/250
51/51 [==============================] - ETA: 0s - loss: 0.9589 - accuracy: 0.7314 - jaccard_coef: 0.4708

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 69s 1s/step - loss: 0.9589 - accuracy: 0.7314 - jaccard_coef: 0.4708 - val_loss: 0.9677 - val_accuracy: 0.7062 - val_jaccard_coef: 0.4801
Epoch 4/250
51/51 [==============================] - 65s 1s/step - loss: 0.9525 - accuracy: 0.7471 - jaccard_coef: 0.5030 - val_loss: 0.9799 - val_accuracy: 0.6515 - val_jaccard_coef: 0.4132
Epoch 5/250
51/51 [==============================] - 64s 1s/step - loss: 0.9492 - accuracy: 0.7525 - jaccard_coef: 0.5163 - val_loss: 0.9729 - val_accuracy: 0.6843 - val_jaccard_coef: 0.4522
Epoch 6/250
51/51 [==============================] - 66s 1s/step - loss: 0.9474 - accuracy: 0.7580 - jaccard_coef: 0.5251 - val_loss: 0.9761 - val_accuracy: 0.6822 - val_jaccard_coef: 0.4639
Epoch 7/250
51/51 [==============================] - 64s 1s/step - loss: 0.9450 - accuracy: 0.7657 - jaccard_coef: 0.5383 - val_loss: 0.9754 - val_accuracy: 0.6937 - val_jaccard_coef: 0.4917
Epoch 8/250
51/51 [==============================] - ETA: 0s - loss: 0.9395 - accuracy: 0.7821 - jaccard_coef: 0.5729

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 68s 1s/step - loss: 0.9395 - accuracy: 0.7821 - jaccard_coef: 0.5729 - val_loss: 0.9619 - val_accuracy: 0.7280 - val_jaccard_coef: 0.5310
Epoch 9/250
51/51 [==============================] - 65s 1s/step - loss: 0.9335 - accuracy: 0.7989 - jaccard_coef: 0.6033 - val_loss: 0.9681 - val_accuracy: 0.7275 - val_jaccard_coef: 0.5402
Epoch 10/250
51/51 [==============================] - ETA: 0s - loss: 0.9328 - accuracy: 0.8020 - jaccard_coef: 0.6049

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 69s 1s/step - loss: 0.9328 - accuracy: 0.8020 - jaccard_coef: 0.6049 - val_loss: 0.9500 - val_accuracy: 0.7626 - val_jaccard_coef: 0.5795
Epoch 11/250
51/51 [==============================] - ETA: 0s - loss: 0.9283 - accuracy: 0.8143 - jaccard_coef: 0.6256

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 67s 1s/step - loss: 0.9283 - accuracy: 0.8143 - jaccard_coef: 0.6256 - val_loss: 0.9472 - val_accuracy: 0.7707 - val_jaccard_coef: 0.5894
Epoch 12/250
51/51 [==============================] - ETA: 0s - loss: 0.9258 - accuracy: 0.8207 - jaccard_coef: 0.6370

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 67s 1s/step - loss: 0.9258 - accuracy: 0.8207 - jaccard_coef: 0.6370 - val_loss: 0.9448 - val_accuracy: 0.7780 - val_jaccard_coef: 0.5977
Epoch 13/250
51/51 [==============================] - ETA: 0s - loss: 0.9233 - accuracy: 0.8267 - jaccard_coef: 0.6472

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 71s 1s/step - loss: 0.9233 - accuracy: 0.8267 - jaccard_coef: 0.6472 - val_loss: 0.9439 - val_accuracy: 0.7799 - val_jaccard_coef: 0.6043
Epoch 14/250
51/51 [==============================] - ETA: 0s - loss: 0.9226 - accuracy: 0.8288 - jaccard_coef: 0.6507

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 72s 1s/step - loss: 0.9226 - accuracy: 0.8288 - jaccard_coef: 0.6507 - val_loss: 0.9409 - val_accuracy: 0.7848 - val_jaccard_coef: 0.6039
Epoch 15/250
51/51 [==============================] - ETA: 0s - loss: 0.9209 - accuracy: 0.8332 - jaccard_coef: 0.6567

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 68s 1s/step - loss: 0.9209 - accuracy: 0.8332 - jaccard_coef: 0.6567 - val_loss: 0.9372 - val_accuracy: 0.7936 - val_jaccard_coef: 0.6254
Epoch 16/250
51/51 [==============================] - ETA: 0s - loss: 0.9189 - accuracy: 0.8370 - jaccard_coef: 0.6654

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 69s 1s/step - loss: 0.9189 - accuracy: 0.8370 - jaccard_coef: 0.6654 - val_loss: 0.9350 - val_accuracy: 0.7971 - val_jaccard_coef: 0.6288
Epoch 17/250
51/51 [==============================] - ETA: 0s - loss: 0.9175 - accuracy: 0.8410 - jaccard_coef: 0.6730

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 68s 1s/step - loss: 0.9175 - accuracy: 0.8410 - jaccard_coef: 0.6730 - val_loss: 0.9334 - val_accuracy: 0.8067 - val_jaccard_coef: 0.6426
Epoch 18/250
51/51 [==============================] - 68s 1s/step - loss: 0.9173 - accuracy: 0.8417 - jaccard_coef: 0.6726 - val_loss: 0.9372 - val_accuracy: 0.7954 - val_jaccard_coef: 0.6252
Epoch 19/250
51/51 [==============================] - 67s 1s/step - loss: 0.9188 - accuracy: 0.8388 - jaccard_coef: 0.6662 - val_loss: 0.9350 - val_accuracy: 0.8011 - val_jaccard_coef: 0.6343
Epoch 20/250
51/51 [==============================] - ETA: 0s - loss: 0.9150 - accuracy: 0.8477 - jaccard_coef: 0.6842

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 73s 1s/step - loss: 0.9150 - accuracy: 0.8477 - jaccard_coef: 0.6842 - val_loss: 0.9318 - val_accuracy: 0.8056 - val_jaccard_coef: 0.6363
Epoch 21/250
51/51 [==============================] - ETA: 0s - loss: 0.9133 - accuracy: 0.8525 - jaccard_coef: 0.6909

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 72s 1s/step - loss: 0.9133 - accuracy: 0.8525 - jaccard_coef: 0.6909 - val_loss: 0.9316 - val_accuracy: 0.8092 - val_jaccard_coef: 0.6484
Epoch 22/250
51/51 [==============================] - ETA: 0s - loss: 0.9120 - accuracy: 0.8561 - jaccard_coef: 0.6983

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 74s 1s/step - loss: 0.9120 - accuracy: 0.8561 - jaccard_coef: 0.6983 - val_loss: 0.9282 - val_accuracy: 0.8174 - val_jaccard_coef: 0.6569
Epoch 23/250
51/51 [==============================] - 67s 1s/step - loss: 0.9112 - accuracy: 0.8585 - jaccard_coef: 0.7013 - val_loss: 0.9311 - val_accuracy: 0.8116 - val_jaccard_coef: 0.6529
Epoch 24/250
51/51 [==============================] - 67s 1s/step - loss: 0.9101 - accuracy: 0.8614 - jaccard_coef: 0.7075 - val_loss: 0.9313 - val_accuracy: 0.8082 - val_jaccard_coef: 0.6457
Epoch 25/250
51/51 [==============================] - 69s 1s/step - loss: 0.9089 - accuracy: 0.8638 - jaccard_coef: 0.7124 - val_loss: 0.9321 - val_accuracy: 0.8147 - val_jaccard_coef: 0.6608
Epoch 26/250
51/51 [==============================] - 69s 1s/step - loss: 0.9080 - accuracy: 0.8651 - jaccard_coef: 0.7187 - val_loss: 0.9344 - val_accuracy: 0.8045 - val_jaccard_coef: 0.6477
Epoch 27/250
51/51 [==============================] - ETA: 0s - loss: 0.9079 - accuracy: 0.8652 - jaccard_coef: 0.7201

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 72s 1s/step - loss: 0.9079 - accuracy: 0.8652 - jaccard_coef: 0.7201 - val_loss: 0.9280 - val_accuracy: 0.8171 - val_jaccard_coef: 0.6614
Epoch 28/250
51/51 [==============================] - 70s 1s/step - loss: 0.9074 - accuracy: 0.8651 - jaccard_coef: 0.7224 - val_loss: 0.9344 - val_accuracy: 0.8029 - val_jaccard_coef: 0.6409
Epoch 29/250
51/51 [==============================] - ETA: 0s - loss: 0.9093 - accuracy: 0.8607 - jaccard_coef: 0.7136

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 75s 1s/step - loss: 0.9093 - accuracy: 0.8607 - jaccard_coef: 0.7136 - val_loss: 0.9246 - val_accuracy: 0.8237 - val_jaccard_coef: 0.6636
Epoch 30/250
51/51 [==============================] - ETA: 0s - loss: 0.9071 - accuracy: 0.8649 - jaccard_coef: 0.7223

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 71s 1s/step - loss: 0.9071 - accuracy: 0.8649 - jaccard_coef: 0.7223 - val_loss: 0.9217 - val_accuracy: 0.8394 - val_jaccard_coef: 0.6951
Epoch 31/250
51/51 [==============================] - ETA: 0s - loss: 0.9058 - accuracy: 0.8714 - jaccard_coef: 0.7317

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 71s 1s/step - loss: 0.9058 - accuracy: 0.8714 - jaccard_coef: 0.7317 - val_loss: 0.9164 - val_accuracy: 0.8479 - val_jaccard_coef: 0.7067
Epoch 32/250
51/51 [==============================] - 66s 1s/step - loss: 0.9054 - accuracy: 0.8683 - jaccard_coef: 0.7290 - val_loss: 0.9205 - val_accuracy: 0.8367 - val_jaccard_coef: 0.6891
Epoch 33/250
51/51 [==============================] - 67s 1s/step - loss: 0.9079 - accuracy: 0.8639 - jaccard_coef: 0.7189 - val_loss: 0.9167 - val_accuracy: 0.8462 - val_jaccard_coef: 0.7067
Epoch 34/250
51/51 [==============================] - 67s 1s/step - loss: 0.9034 - accuracy: 0.8741 - jaccard_coef: 0.7376 - val_loss: 0.9182 - val_accuracy: 0.8389 - val_jaccard_coef: 0.7021
Epoch 35/250
51/51 [==============================] - 66s 1s/step - loss: 0.9002 - accuracy: 0.8793 - jaccard_coef: 0.7487 - val_loss: 0.9182 - val_accuracy: 0.8459 - val_jaccard_coef: 0.7115
Epoch 36/250
51/51 [==============================] - 68s 1s/step - loss: 0.8976 - accuracy: 0.8832 - jaccard_coef: 0.7559 - val_loss: 0.9201 - val_accuracy: 0.8376 - val_jaccard_coef: 0.7014
Epoch 37/250
51/51 [==============================] - 68s 1s/step - loss: 0.9013 - accuracy: 0.8777 - jaccard_coef: 0.7433 - val_loss: 0.9228 - val_accuracy: 0.8249 - val_jaccard_coef: 0.6774
Epoch 38/250
51/51 [==============================] - ETA: 0s - loss: 0.8989 - accuracy: 0.8816 - jaccard_coef: 0.7523

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 74s 1s/step - loss: 0.8989 - accuracy: 0.8816 - jaccard_coef: 0.7523 - val_loss: 0.9154 - val_accuracy: 0.8407 - val_jaccard_coef: 0.6942
Epoch 39/250
51/51 [==============================] - 70s 1s/step - loss: 0.8972 - accuracy: 0.8841 - jaccard_coef: 0.7558 - val_loss: 0.9225 - val_accuracy: 0.8305 - val_jaccard_coef: 0.6881
Epoch 40/250
51/51 [==============================] - 69s 1s/step - loss: 0.8961 - accuracy: 0.8844 - jaccard_coef: 0.7576 - val_loss: 0.9329 - val_accuracy: 0.8125 - val_jaccard_coef: 0.6649
Epoch 41/250
51/51 [==============================] - 65s 1s/step - loss: 0.8980 - accuracy: 0.8823 - jaccard_coef: 0.7525 - val_loss: 0.9223 - val_accuracy: 0.8305 - val_jaccard_coef: 0.6870
Epoch 42/250
51/51 [==============================] - 65s 1s/step - loss: 0.9008 - accuracy: 0.8726 - jaccard_coef: 0.7371 - val_loss: 0.9256 - val_accuracy: 0.8231 - val_jaccard_coef: 0.6676
Epoch 43/250
51/51 [==============================] - 65s 1s/step - loss: 0.9127 - accuracy: 0.8473 - jaccard_coef: 0.6915 - val_loss: 0.9216 - val_accuracy: 0.8365 - val_jaccard_coef: 0.6928
Epoch 44/250
51/51 [==============================] - 64s 1s/step - loss: 0.9011 - accuracy: 0.8766 - jaccard_coef: 0.7404 - val_loss: 0.9180 - val_accuracy: 0.8385 - val_jaccard_coef: 0.6904
Epoch 45/250
51/51 [==============================] - 64s 1s/step - loss: 0.8969 - accuracy: 0.8829 - jaccard_coef: 0.7543 - val_loss: 0.9181 - val_accuracy: 0.8459 - val_jaccard_coef: 0.7138
Epoch 46/250
51/51 [==============================] - ETA: 0s - loss: 0.8947 - accuracy: 0.8871 - jaccard_coef: 0.7621

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 67s 1s/step - loss: 0.8947 - accuracy: 0.8871 - jaccard_coef: 0.7621 - val_loss: 0.9134 - val_accuracy: 0.8519 - val_jaccard_coef: 0.7214
Epoch 47/250
51/51 [==============================] - 64s 1s/step - loss: 0.8953 - accuracy: 0.8859 - jaccard_coef: 0.7611 - val_loss: 0.9166 - val_accuracy: 0.8455 - val_jaccard_coef: 0.7114
Epoch 48/250
51/51 [==============================] - ETA: 0s - loss: 0.8939 - accuracy: 0.8896 - jaccard_coef: 0.7656

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 68s 1s/step - loss: 0.8939 - accuracy: 0.8896 - jaccard_coef: 0.7656 - val_loss: 0.9126 - val_accuracy: 0.8536 - val_jaccard_coef: 0.7215
Epoch 49/250
51/51 [==============================] - 64s 1s/step - loss: 0.8920 - accuracy: 0.8941 - jaccard_coef: 0.7756 - val_loss: 0.9149 - val_accuracy: 0.8513 - val_jaccard_coef: 0.7237
Epoch 50/250
51/51 [==============================] - 64s 1s/step - loss: 0.8899 - accuracy: 0.8980 - jaccard_coef: 0.7832 - val_loss: 0.9157 - val_accuracy: 0.8578 - val_jaccard_coef: 0.7318
Epoch 51/250
51/51 [==============================] - 64s 1s/step - loss: 0.8910 - accuracy: 0.8955 - jaccard_coef: 0.7782 - val_loss: 0.9159 - val_accuracy: 0.8586 - val_jaccard_coef: 0.7362
Epoch 52/250
51/51 [==============================] - 64s 1s/step - loss: 0.8899 - accuracy: 0.8978 - jaccard_coef: 0.7826 - val_loss: 0.9156 - val_accuracy: 0.8524 - val_jaccard_coef: 0.7229
Epoch 53/250
51/51 [==============================] - 65s 1s/step - loss: 0.8923 - accuracy: 0.8939 - jaccard_coef: 0.7738 - val_loss: 0.9298 - val_accuracy: 0.8220 - val_jaccard_coef: 0.6722
Epoch 54/250
51/51 [==============================] - 65s 1s/step - loss: 0.8985 - accuracy: 0.8796 - jaccard_coef: 0.7489 - val_loss: 0.9142 - val_accuracy: 0.8498 - val_jaccard_coef: 0.7141
Epoch 55/250
51/51 [==============================] - 69s 1s/step - loss: 0.8948 - accuracy: 0.8886 - jaccard_coef: 0.7644 - val_loss: 0.9135 - val_accuracy: 0.8548 - val_jaccard_coef: 0.7263
Epoch 56/250
51/51 [==============================] - 71s 1s/step - loss: 0.8924 - accuracy: 0.8916 - jaccard_coef: 0.7709 - val_loss: 0.9135 - val_accuracy: 0.8523 - val_jaccard_coef: 0.7184
Epoch 57/250
51/51 [==============================] - 69s 1s/step - loss: 0.8899 - accuracy: 0.8976 - jaccard_coef: 0.7808 - val_loss: 0.9220 - val_accuracy: 0.8245 - val_jaccard_coef: 0.6760
Epoch 58/250
51/51 [==============================] - 68s 1s/step - loss: 0.8930 - accuracy: 0.8891 - jaccard_coef: 0.7662 - val_loss: 0.9194 - val_accuracy: 0.8507 - val_jaccard_coef: 0.7201
Epoch 59/250
51/51 [==============================] - 69s 1s/step - loss: 0.8896 - accuracy: 0.8983 - jaccard_coef: 0.7826 - val_loss: 0.9168 - val_accuracy: 0.8534 - val_jaccard_coef: 0.7269
Epoch 60/250
51/51 [==============================] - 66s 1s/step - loss: 0.8874 - accuracy: 0.9021 - jaccard_coef: 0.7909 - val_loss: 0.9139 - val_accuracy: 0.8567 - val_jaccard_coef: 0.7301
Epoch 61/250
51/51 [==============================] - ETA: 0s - loss: 0.8863 - accuracy: 0.9058 - jaccard_coef: 0.7984

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 69s 1s/step - loss: 0.8863 - accuracy: 0.9058 - jaccard_coef: 0.7984 - val_loss: 0.9107 - val_accuracy: 0.8606 - val_jaccard_coef: 0.7364
Epoch 62/250
51/51 [==============================] - 68s 1s/step - loss: 0.8873 - accuracy: 0.9021 - jaccard_coef: 0.7907 - val_loss: 0.9148 - val_accuracy: 0.8552 - val_jaccard_coef: 0.7285
Epoch 63/250
51/51 [==============================] - 67s 1s/step - loss: 0.8851 - accuracy: 0.9070 - jaccard_coef: 0.8007 - val_loss: 0.9123 - val_accuracy: 0.8606 - val_jaccard_coef: 0.7366
Epoch 64/250
51/51 [==============================] - 65s 1s/step - loss: 0.8845 - accuracy: 0.9065 - jaccard_coef: 0.8006 - val_loss: 0.9176 - val_accuracy: 0.8545 - val_jaccard_coef: 0.7296
Epoch 65/250
51/51 [==============================] - 65s 1s/step - loss: 0.8839 - accuracy: 0.9088 - jaccard_coef: 0.8039 - val_loss: 0.9151 - val_accuracy: 0.8569 - val_jaccard_coef: 0.7318
Epoch 66/250
51/51 [==============================] - 65s 1s/step - loss: 0.8833 - accuracy: 0.9102 - jaccard_coef: 0.8071 - val_loss: 0.9129 - val_accuracy: 0.8647 - val_jaccard_coef: 0.7424
Epoch 67/250
51/51 [==============================] - 66s 1s/step - loss: 0.8836 - accuracy: 0.9094 - jaccard_coef: 0.8058 - val_loss: 0.9141 - val_accuracy: 0.8596 - val_jaccard_coef: 0.7346
Epoch 68/250
51/51 [==============================] - ETA: 0s - loss: 0.8828 - accuracy: 0.9107 - jaccard_coef: 0.8081

# ===== Cell Separator =====

/home/user/.local/lib/python3.10/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

# ===== Cell Separator =====

INFO:tensorflow:Assets written to: /home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best/assets
wandb: Adding directory to artifact (/home/user/wandb/run-20251006_102958-kq2bngeb/files/model-best)... Done. 0.1s

# ===== Cell Separator =====

51/51 [==============================] - 70s 1s/step - loss: 0.8828 - accuracy: 0.9107 - jaccard_coef: 0.8081 - val_loss: 0.9107 - val_accuracy: 0.8659 - val_jaccard_coef: 0.7455
Epoch 69/250
51/51 [==============================] - 68s 1s/step - loss: 0.8842 - accuracy: 0.9088 - jaccard_coef: 0.8049 - val_loss: 0.9109 - val_accuracy: 0.8635 - val_jaccard_coef: 0.7418
Epoch 70/250
51/51 [==============================] - 70s 1s/step - loss: 0.8830 - accuracy: 0.9106 - jaccard_coef: 0.8083 - val_loss: 0.9137 - val_accuracy: 0.8612 - val_jaccard_coef: 0.7379
Epoch 71/250
51/51 [==============================] - 72s 1s/step - loss: 0.8817 - accuracy: 0.9144 - jaccard_coef: 0.8152 - val_loss: 0.9142 - val_accuracy: 0.8625 - val_jaccard_coef: 0.7419
Epoch 72/250
51/51 [==============================] - 70s 1s/step - loss: 0.8798 - accuracy: 0.9183 - jaccard_coef: 0.8233 - val_loss: 0.9153 - val_accuracy: 0.8608 - val_jaccard_coef: 0.7397
Epoch 73/250
51/51 [==============================] - 70s 1s/step - loss: 0.8792 - accuracy: 0.9192 - jaccard_coef: 0.8250 - val_loss: 0.9142 - val_accuracy: 0.8601 - val_jaccard_coef: 0.7384
Epoch 74/250
51/51 [==============================] - 70s 1s/step - loss: 0.8790 - accuracy: 0.9188 - jaccard_coef: 0.8245 - val_loss: 0.9185 - val_accuracy: 0.8559 - val_jaccard_coef: 0.7305
Epoch 75/250
51/51 [==============================] - 69s 1s/step - loss: 0.8786 - accuracy: 0.9202 - jaccard_coef: 0.8270 - val_loss: 0.9135 - val_accuracy: 0.8649 - val_jaccard_coef: 0.7462
Epoch 76/250
51/51 [==============================] - 69s 1s/step - loss: 0.8789 - accuracy: 0.9196 - jaccard_coef: 0.8263 - val_loss: 0.9151 - val_accuracy: 0.8599 - val_jaccard_coef: 0.7384
Epoch 77/250
51/51 [==============================] - 61s 1s/step - loss: 0.8790 - accuracy: 0.9186 - jaccard_coef: 0.8240 - val_loss: 0.9164 - val_accuracy: 0.8562 - val_jaccard_coef: 0.7343
Epoch 78/250
51/51 [==============================] - 61s 1s/step - loss: 0.8776 - accuracy: 0.9220 - jaccard_coef: 0.8309 - val_loss: 0.9236 - val_accuracy: 0.8449 - val_jaccard_coef: 0.7178
Epoch 79/250
51/51 [==============================] - 61s 1s/step - loss: 0.8784 - accuracy: 0.9208 - jaccard_coef: 0.8289 - val_loss: 0.9193 - val_accuracy: 0.8522 - val_jaccard_coef: 0.7284
Epoch 80/250
51/51 [==============================] - 61s 1s/step - loss: 0.8765 - accuracy: 0.9243 - jaccard_coef: 0.8353 - val_loss: 0.9165 - val_accuracy: 0.8593 - val_jaccard_coef: 0.7399
Epoch 81/250
51/51 [==============================] - 66s 1s/step - loss: 0.8775 - accuracy: 0.9220 - jaccard_coef: 0.8309 - val_loss: 0.9162 - val_accuracy: 0.8595 - val_jaccard_coef: 0.7391
Epoch 82/250
51/51 [==============================] - 69s 1s/step - loss: 0.8767 - accuracy: 0.9231 - jaccard_coef: 0.8328 - val_loss: 0.9207 - val_accuracy: 0.8553 - val_jaccard_coef: 0.7335
Epoch 83/250
51/51 [==============================] - 70s 1s/step - loss: 0.8760 - accuracy: 0.9249 - jaccard_coef: 0.8367 - val_loss: 0.9256 - val_accuracy: 0.8457 - val_jaccard_coef: 0.7205
Epoch 84/250
51/51 [==============================] - 68s 1s/step - loss: 0.8762 - accuracy: 0.9236 - jaccard_coef: 0.8342 - val_loss: 0.9281 - val_accuracy: 0.8370 - val_jaccard_coef: 0.7063
Epoch 85/250
51/51 [==============================] - 69s 1s/step - loss: 0.8777 - accuracy: 0.9200 - jaccard_coef: 0.8274 - val_loss: 0.9183 - val_accuracy: 0.8503 - val_jaccard_coef: 0.7122
Epoch 86/250
51/51 [==============================] - 71s 1s/step - loss: 0.8768 - accuracy: 0.9220 - jaccard_coef: 0.8311 - val_loss: 0.9244 - val_accuracy: 0.8449 - val_jaccard_coef: 0.7181
Epoch 87/250
51/51 [==============================] - 70s 1s/step - loss: 0.8780 - accuracy: 0.9189 - jaccard_coef: 0.8260 - val_loss: 0.9277 - val_accuracy: 0.8346 - val_jaccard_coef: 0.7010
Epoch 88/250
51/51 [==============================] - 70s 1s/step - loss: 0.8804 - accuracy: 0.9142 - jaccard_coef: 0.8149 - val_loss: 0.9222 - val_accuracy: 0.8457 - val_jaccard_coef: 0.7116
Epoch 89/250
51/51 [==============================] - 68s 1s/step - loss: 0.8787 - accuracy: 0.9168 - jaccard_coef: 0.8206 - val_loss: 0.9263 - val_accuracy: 0.8380 - val_jaccard_coef: 0.7067
Epoch 90/250
51/51 [==============================] - 70s 1s/step - loss: 0.8764 - accuracy: 0.9232 - jaccard_coef: 0.8331 - val_loss: 0.9294 - val_accuracy: 0.8333 - val_jaccard_coef: 0.7032
Epoch 91/250
51/51 [==============================] - 70s 1s/step - loss: 0.8778 - accuracy: 0.9188 - jaccard_coef: 0.8252 - val_loss: 0.9289 - val_accuracy: 0.8356 - val_jaccard_coef: 0.7062
Epoch 92/250
51/51 [==============================] - 68s 1s/step - loss: 0.8765 - accuracy: 0.9224 - jaccard_coef: 0.8321 - val_loss: 0.9200 - val_accuracy: 0.8519 - val_jaccard_coef: 0.7289
Epoch 93/250
51/51 [==============================] - 69s 1s/step - loss: 0.8764 - accuracy: 0.9223 - jaccard_coef: 0.8316 - val_loss: 0.9209 - val_accuracy: 0.8533 - val_jaccard_coef: 0.7311
Epoch 94/250
51/51 [==============================] - 65s 1s/step - loss: 0.8768 - accuracy: 0.9216 - jaccard_coef: 0.8301 - val_loss: 0.9202 - val_accuracy: 0.8511 - val_jaccard_coef: 0.7280
Epoch 95/250
51/51 [==============================] - 69s 1s/step - loss: 0.8765 - accuracy: 0.9234 - jaccard_coef: 0.8335 - val_loss: 0.9178 - val_accuracy: 0.8560 - val_jaccard_coef: 0.7364
Epoch 96/250
51/51 [==============================] - 68s 1s/step - loss: 0.8733 - accuracy: 0.9300 - jaccard_coef: 0.8471 - val_loss: 0.9186 - val_accuracy: 0.8581 - val_jaccard_coef: 0.7388
Epoch 97/250
51/51 [==============================] - 67s 1s/step - loss: 0.8735 - accuracy: 0.9297 - jaccard_coef: 0.8466 - val_loss: 0.9169 - val_accuracy: 0.8570 - val_jaccard_coef: 0.7381
Epoch 98/250
51/51 [==============================] - 65s 1s/step - loss: 0.8720 - accuracy: 0.9331 - jaccard_coef: 0.8529 - val_loss: 0.9172 - val_accuracy: 0.8564 - val_jaccard_coef: 0.7364
Epoch 99/250
51/51 [==============================] - 63s 1s/step - loss: 0.8721 - accuracy: 0.9337 - jaccard_coef: 0.8540 - val_loss: 0.9214 - val_accuracy: 0.8530 - val_jaccard_coef: 0.7339
Epoch 100/250
51/51 [==============================] - 64s 1s/step - loss: 0.8711 - accuracy: 0.9352 - jaccard_coef: 0.8579 - val_loss: 0.9144 - val_accuracy: 0.8600 - val_jaccard_coef: 0.7418
Epoch 101/250
51/51 [==============================] - 63s 1s/step - loss: 0.8718 - accuracy: 0.9347 - jaccard_coef: 0.8566 - val_loss: 0.9175 - val_accuracy: 0.8558 - val_jaccard_coef: 0.7367
Epoch 102/250
51/51 [==============================] - 68s 1s/step - loss: 0.8716 - accuracy: 0.9351 - jaccard_coef: 0.8575 - val_loss: 0.9215 - val_accuracy: 0.8544 - val_jaccard_coef: 0.7358
Epoch 103/250
51/51 [==============================] - 64s 1s/step - loss: 0.8708 - accuracy: 0.9357 - jaccard_coef: 0.8587 - val_loss: 0.9173 - val_accuracy: 0.8622 - val_jaccard_coef: 0.7463
Epoch 104/250
51/51 [==============================] - 68s 1s/step - loss: 0.8737 - accuracy: 0.9291 - jaccard_coef: 0.8452 - val_loss: 0.9125 - val_accuracy: 0.8626 - val_jaccard_coef: 0.7440
Epoch 105/250
51/51 [==============================] - 73s 1s/step - loss: 0.8718 - accuracy: 0.9332 - jaccard_coef: 0.8534 - val_loss: 0.9162 - val_accuracy: 0.8589 - val_jaccard_coef: 0.7398
Epoch 106/250
51/51 [==============================] - 66s 1s/step - loss: 0.8710 - accuracy: 0.9356 - jaccard_coef: 0.8587 - val_loss: 0.9208 - val_accuracy: 0.8529 - val_jaccard_coef: 0.7329
Epoch 107/250
51/51 [==============================] - 69s 1s/step - loss: 0.8706 - accuracy: 0.9368 - jaccard_coef: 0.8608 - val_loss: 0.9213 - val_accuracy: 0.8520 - val_jaccard_coef: 0.7310
Epoch 108/250
51/51 [==============================] - 67s 1s/step - loss: 0.8698 - accuracy: 0.9385 - jaccard_coef: 0.8645 - val_loss: 0.9218 - val_accuracy: 0.8534 - val_jaccard_coef: 0.7342
Epoch 109/250
51/51 [==============================] - 67s 1s/step - loss: 0.8696 - accuracy: 0.9385 - jaccard_coef: 0.8646 - val_loss: 0.9215 - val_accuracy: 0.8561 - val_jaccard_coef: 0.7385
Epoch 110/250
51/51 [==============================] - 66s 1s/step - loss: 0.8695 - accuracy: 0.9390 - jaccard_coef: 0.8659 - val_loss: 0.9176 - val_accuracy: 0.8605 - val_jaccard_coef: 0.7432
Epoch 111/250
51/51 [==============================] - 64s 1s/step - loss: 0.8693 - accuracy: 0.9399 - jaccard_coef: 0.8673 - val_loss: 0.9202 - val_accuracy: 0.8573 - val_jaccard_coef: 0.7392
Epoch 112/250
51/51 [==============================] - 63s 1s/step - loss: 0.8687 - accuracy: 0.9410 - jaccard_coef: 0.8698 - val_loss: 0.9235 - val_accuracy: 0.8521 - val_jaccard_coef: 0.7330
Epoch 113/250
51/51 [==============================] - 64s 1s/step - loss: 0.8688 - accuracy: 0.9407 - jaccard_coef: 0.8693 - val_loss: 0.9215 - val_accuracy: 0.8555 - val_jaccard_coef: 0.7376
Epoch 114/250
51/51 [==============================] - 64s 1s/step - loss: 0.8696 - accuracy: 0.9389 - jaccard_coef: 0.8657 - val_loss: 0.9160 - val_accuracy: 0.8633 - val_jaccard_coef: 0.7480
Epoch 115/250
51/51 [==============================] - 63s 1s/step - loss: 0.8698 - accuracy: 0.9390 - jaccard_coef: 0.8656 - val_loss: 0.9249 - val_accuracy: 0.8499 - val_jaccard_coef: 0.7293
Epoch 116/250
51/51 [==============================] - 66s 1s/step - loss: 0.8704 - accuracy: 0.9357 - jaccard_coef: 0.8590 - val_loss: 0.9207 - val_accuracy: 0.8546 - val_jaccard_coef: 0.7355
Epoch 117/250
51/51 [==============================] - 63s 1s/step - loss: 0.8689 - accuracy: 0.9402 - jaccard_coef: 0.8680 - val_loss: 0.9218 - val_accuracy: 0.8569 - val_jaccard_coef: 0.7399
Epoch 118/250
51/51 [==============================] - 64s 1s/step - loss: 0.8687 - accuracy: 0.9401 - jaccard_coef: 0.8678 - val_loss: 0.9200 - val_accuracy: 0.8574 - val_jaccard_coef: 0.7396
Epoch 119/250
51/51 [==============================] - 63s 1s/step - loss: 0.8682 - accuracy: 0.9420 - jaccard_coef: 0.8719 - val_loss: 0.9207 - val_accuracy: 0.8599 - val_jaccard_coef: 0.7436
Epoch 120/250
51/51 [==============================] - 63s 1s/step - loss: 0.8694 - accuracy: 0.9408 - jaccard_coef: 0.8690 - val_loss: 0.9298 - val_accuracy: 0.8461 - val_jaccard_coef: 0.7232
Epoch 121/250
51/51 [==============================] - 71s 1s/step - loss: 0.8796 - accuracy: 0.9157 - jaccard_coef: 0.8192 - val_loss: 0.9269 - val_accuracy: 0.8341 - val_jaccard_coef: 0.6901
Epoch 122/250
51/51 [==============================] - 66s 1s/step - loss: 0.8918 - accuracy: 0.8893 - jaccard_coef: 0.7684 - val_loss: 0.9199 - val_accuracy: 0.8391 - val_jaccard_coef: 0.6940
Epoch 123/250
51/51 [==============================] - 67s 1s/step - loss: 0.8847 - accuracy: 0.9048 - jaccard_coef: 0.7974 - val_loss: 0.9196 - val_accuracy: 0.8542 - val_jaccard_coef: 0.7274
Epoch 124/250
51/51 [==============================] - 69s 1s/step - loss: 0.8776 - accuracy: 0.9201 - jaccard_coef: 0.8268 - val_loss: 0.9217 - val_accuracy: 0.8490 - val_jaccard_coef: 0.7242
Epoch 125/250
51/51 [==============================] - 67s 1s/step - loss: 0.8771 - accuracy: 0.9229 - jaccard_coef: 0.8332 - val_loss: 0.9274 - val_accuracy: 0.8423 - val_jaccard_coef: 0.7155
Epoch 126/250
51/51 [==============================] - 65s 1s/step - loss: 0.8748 - accuracy: 0.9255 - jaccard_coef: 0.8384 - val_loss: 0.9167 - val_accuracy: 0.8531 - val_jaccard_coef: 0.7309
Epoch 127/250
51/51 [==============================] - 67s 1s/step - loss: 0.8713 - accuracy: 0.9349 - jaccard_coef: 0.8569 - val_loss: 0.9160 - val_accuracy: 0.8628 - val_jaccard_coef: 0.7477
Epoch 128/250
51/51 [==============================] - 72s 1s/step - loss: 0.8705 - accuracy: 0.9359 - jaccard_coef: 0.8587 - val_loss: 0.9189 - val_accuracy: 0.8559 - val_jaccard_coef: 0.7375
Epoch 129/250
51/51 [==============================] - 71s 1s/step - loss: 0.8690 - accuracy: 0.9397 - jaccard_coef: 0.8670 - val_loss: 0.9214 - val_accuracy: 0.8533 - val_jaccard_coef: 0.7343
Epoch 130/250
51/51 [==============================] - 67s 1s/step - loss: 0.8684 - accuracy: 0.9411 - jaccard_coef: 0.8699 - val_loss: 0.9200 - val_accuracy: 0.8585 - val_jaccard_coef: 0.7420
Epoch 131/250
51/51 [==============================] - 67s 1s/step - loss: 0.8677 - accuracy: 0.9430 - jaccard_coef: 0.8737 - val_loss: 0.9182 - val_accuracy: 0.8606 - val_jaccard_coef: 0.7466
Epoch 132/250
51/51 [==============================] - 69s 1s/step - loss: 0.8674 - accuracy: 0.9437 - jaccard_coef: 0.8754 - val_loss: 0.9222 - val_accuracy: 0.8545 - val_jaccard_coef: 0.7374
Epoch 133/250
51/51 [==============================] - 71s 1s/step - loss: 0.8672 - accuracy: 0.9440 - jaccard_coef: 0.8763 - val_loss: 0.9200 - val_accuracy: 0.8564 - val_jaccard_coef: 0.7387
Epoch 134/250
51/51 [==============================] - 68s 1s/step - loss: 0.8671 - accuracy: 0.9444 - jaccard_coef: 0.8769 - val_loss: 0.9244 - val_accuracy: 0.8512 - val_jaccard_coef: 0.7317
Epoch 135/250
51/51 [==============================] - 66s 1s/step - loss: 0.8689 - accuracy: 0.9390 - jaccard_coef: 0.8658 - val_loss: 0.9213 - val_accuracy: 0.8485 - val_jaccard_coef: 0.7248
Epoch 136/250
51/51 [==============================] - 62s 1s/step - loss: 0.8673 - accuracy: 0.9436 - jaccard_coef: 0.8749 - val_loss: 0.9237 - val_accuracy: 0.8500 - val_jaccard_coef: 0.7300
Epoch 137/250
51/51 [==============================] - 61s 1s/step - loss: 0.8669 - accuracy: 0.9443 - jaccard_coef: 0.8768 - val_loss: 0.9210 - val_accuracy: 0.8544 - val_jaccard_coef: 0.7358
Epoch 138/250
51/51 [==============================] - 62s 1s/step - loss: 0.8665 - accuracy: 0.9453 - jaccard_coef: 0.8791 - val_loss: 0.9227 - val_accuracy: 0.8528 - val_jaccard_coef: 0.7340
Epoch 139/250
51/51 [==============================] - 61s 1s/step - loss: 0.8667 - accuracy: 0.9452 - jaccard_coef: 0.8785 - val_loss: 0.9239 - val_accuracy: 0.8470 - val_jaccard_coef: 0.7243
Epoch 140/250
51/51 [==============================] - 61s 1s/step - loss: 0.8664 - accuracy: 0.9457 - jaccard_coef: 0.8796 - val_loss: 0.9171 - val_accuracy: 0.8585 - val_jaccard_coef: 0.7423
Epoch 141/250
51/51 [==============================] - 61s 1s/step - loss: 0.8663 - accuracy: 0.9456 - jaccard_coef: 0.8793 - val_loss: 0.9200 - val_accuracy: 0.8547 - val_jaccard_coef: 0.7365
Epoch 142/250
51/51 [==============================] - 61s 1s/step - loss: 0.8664 - accuracy: 0.9453 - jaccard_coef: 0.8785 - val_loss: 0.9214 - val_accuracy: 0.8521 - val_jaccard_coef: 0.7318
Epoch 143/250
51/51 [==============================] - 61s 1s/step - loss: 0.8668 - accuracy: 0.9442 - jaccard_coef: 0.8765 - val_loss: 0.9190 - val_accuracy: 0.8565 - val_jaccard_coef: 0.7391
Epoch 144/250
51/51 [==============================] - 61s 1s/step - loss: 0.8660 - accuracy: 0.9465 - jaccard_coef: 0.8814 - val_loss: 0.9210 - val_accuracy: 0.8535 - val_jaccard_coef: 0.7340
Epoch 145/250
51/51 [==============================] - 62s 1s/step - loss: 0.8672 - accuracy: 0.9437 - jaccard_coef: 0.8757 - val_loss: 0.9173 - val_accuracy: 0.8642 - val_jaccard_coef: 0.7509
Epoch 146/250
51/51 [==============================] - 61s 1s/step - loss: 0.8659 - accuracy: 0.9464 - jaccard_coef: 0.8811 - val_loss: 0.9204 - val_accuracy: 0.8560 - val_jaccard_coef: 0.7386
Epoch 147/250
51/51 [==============================] - 61s 1s/step - loss: 0.8724 - accuracy: 0.9319 - jaccard_coef: 0.8520 - val_loss: 0.9162 - val_accuracy: 0.8484 - val_jaccard_coef: 0.7182
Epoch 148/250
51/51 [==============================] - 62s 1s/step - loss: 0.8698 - accuracy: 0.9378 - jaccard_coef: 0.8627 - val_loss: 0.9158 - val_accuracy: 0.8624 - val_jaccard_coef: 0.7470
Epoch 149/250
51/51 [==============================] - 61s 1s/step - loss: 0.8664 - accuracy: 0.9458 - jaccard_coef: 0.8793 - val_loss: 0.9180 - val_accuracy: 0.8592 - val_jaccard_coef: 0.7423
Epoch 150/250
51/51 [==============================] - 61s 1s/step - loss: 0.8653 - accuracy: 0.9481 - jaccard_coef: 0.8841 - val_loss: 0.9139 - val_accuracy: 0.8667 - val_jaccard_coef: 0.7552
Epoch 151/250
51/51 [==============================] - 61s 1s/step - loss: 0.8651 - accuracy: 0.9486 - jaccard_coef: 0.8855 - val_loss: 0.9168 - val_accuracy: 0.8642 - val_jaccard_coef: 0.7516
Epoch 152/250
51/51 [==============================] - 61s 1s/step - loss: 0.8651 - accuracy: 0.9486 - jaccard_coef: 0.8858 - val_loss: 0.9134 - val_accuracy: 0.8686 - val_jaccard_coef: 0.7572
Epoch 153/250
51/51 [==============================] - 61s 1s/step - loss: 0.8649 - accuracy: 0.9492 - jaccard_coef: 0.8870 - val_loss: 0.9142 - val_accuracy: 0.8676 - val_jaccard_coef: 0.7567
Epoch 154/250
51/51 [==============================] - 61s 1s/step - loss: 0.8647 - accuracy: 0.9493 - jaccard_coef: 0.8875 - val_loss: 0.9166 - val_accuracy: 0.8659 - val_jaccard_coef: 0.7540
Epoch 155/250
51/51 [==============================] - 61s 1s/step - loss: 0.8647 - accuracy: 0.9494 - jaccard_coef: 0.8876 - val_loss: 0.9204 - val_accuracy: 0.8601 - val_jaccard_coef: 0.7458
Epoch 156/250
51/51 [==============================] - 61s 1s/step - loss: 0.8651 - accuracy: 0.9483 - jaccard_coef: 0.8852 - val_loss: 0.9235 - val_accuracy: 0.8541 - val_jaccard_coef: 0.7361
Epoch 157/250
51/51 [==============================] - 61s 1s/step - loss: 0.8649 - accuracy: 0.9488 - jaccard_coef: 0.8859 - val_loss: 0.9270 - val_accuracy: 0.8529 - val_jaccard_coef: 0.7361
Epoch 158/250
51/51 [==============================] - 61s 1s/step - loss: 0.8650 - accuracy: 0.9496 - jaccard_coef: 0.8879 - val_loss: 0.9290 - val_accuracy: 0.8462 - val_jaccard_coef: 0.7231
Epoch 159/250
51/51 [==============================] - 62s 1s/step - loss: 0.8648 - accuracy: 0.9494 - jaccard_coef: 0.8875 - val_loss: 0.9308 - val_accuracy: 0.8424 - val_jaccard_coef: 0.7206
Epoch 160/250
51/51 [==============================] - 61s 1s/step - loss: 0.8647 - accuracy: 0.9494 - jaccard_coef: 0.8874 - val_loss: 0.9317 - val_accuracy: 0.8467 - val_jaccard_coef: 0.7261
Epoch 161/250
51/51 [==============================] - 61s 1s/step - loss: 0.8647 - accuracy: 0.9494 - jaccard_coef: 0.8873 - val_loss: 0.9226 - val_accuracy: 0.8512 - val_jaccard_coef: 0.7326
Epoch 162/250
51/51 [==============================] - 61s 1s/step - loss: 0.8651 - accuracy: 0.9484 - jaccard_coef: 0.8854 - val_loss: 0.9304 - val_accuracy: 0.8469 - val_jaccard_coef: 0.7276
Epoch 163/250
51/51 [==============================] - 61s 1s/step - loss: 0.8643 - accuracy: 0.9503 - jaccard_coef: 0.8891 - val_loss: 0.9281 - val_accuracy: 0.8497 - val_jaccard_coef: 0.7313
Epoch 164/250
51/51 [==============================] - 61s 1s/step - loss: 0.8643 - accuracy: 0.9504 - jaccard_coef: 0.8894 - val_loss: 0.9284 - val_accuracy: 0.8523 - val_jaccard_coef: 0.7352
Epoch 165/250
51/51 [==============================] - 61s 1s/step - loss: 0.8648 - accuracy: 0.9487 - jaccard_coef: 0.8863 - val_loss: 0.9232 - val_accuracy: 0.8559 - val_jaccard_coef: 0.7388
Epoch 166/250
51/51 [==============================] - 61s 1s/step - loss: 0.8642 - accuracy: 0.9496 - jaccard_coef: 0.8876 - val_loss: 0.9254 - val_accuracy: 0.8560 - val_jaccard_coef: 0.7405
Epoch 167/250
51/51 [==============================] - 61s 1s/step - loss: 0.8634 - accuracy: 0.9516 - jaccard_coef: 0.8919 - val_loss: 0.9210 - val_accuracy: 0.8621 - val_jaccard_coef: 0.7498
Epoch 168/250
51/51 [==============================] - 61s 1s/step - loss: 0.8632 - accuracy: 0.9526 - jaccard_coef: 0.8940 - val_loss: 0.9174 - val_accuracy: 0.8657 - val_jaccard_coef: 0.7540
Epoch 169/250
51/51 [==============================] - 61s 1s/step - loss: 0.8633 - accuracy: 0.9526 - jaccard_coef: 0.8940 - val_loss: 0.9193 - val_accuracy: 0.8655 - val_jaccard_coef: 0.7543
Epoch 170/250
51/51 [==============================] - 61s 1s/step - loss: 0.8629 - accuracy: 0.9533 - jaccard_coef: 0.8958 - val_loss: 0.9215 - val_accuracy: 0.8608 - val_jaccard_coef: 0.7473
Epoch 171/250
51/51 [==============================] - 61s 1s/step - loss: 0.8631 - accuracy: 0.9530 - jaccard_coef: 0.8950 - val_loss: 0.9226 - val_accuracy: 0.8574 - val_jaccard_coef: 0.7423
Epoch 172/250
51/51 [==============================] - 61s 1s/step - loss: 0.8643 - accuracy: 0.9501 - jaccard_coef: 0.8887 - val_loss: 0.9202 - val_accuracy: 0.8608 - val_jaccard_coef: 0.7469
Epoch 173/250
51/51 [==============================] - 61s 1s/step - loss: 0.8636 - accuracy: 0.9517 - jaccard_coef: 0.8921 - val_loss: 0.9212 - val_accuracy: 0.8588 - val_jaccard_coef: 0.7442
Epoch 174/250
51/51 [==============================] - 61s 1s/step - loss: 0.8626 - accuracy: 0.9536 - jaccard_coef: 0.8959 - val_loss: 0.9259 - val_accuracy: 0.8565 - val_jaccard_coef: 0.7418
Epoch 175/250
51/51 [==============================] - 61s 1s/step - loss: 0.8628 - accuracy: 0.9531 - jaccard_coef: 0.8950 - val_loss: 0.9250 - val_accuracy: 0.8571 - val_jaccard_coef: 0.7425
Epoch 176/250
51/51 [==============================] - 62s 1s/step - loss: 0.8623 - accuracy: 0.9540 - jaccard_coef: 0.8971 - val_loss: 0.9303 - val_accuracy: 0.8480 - val_jaccard_coef: 0.7297
Epoch 177/250
51/51 [==============================] - 62s 1s/step - loss: 0.8623 - accuracy: 0.9544 - jaccard_coef: 0.8980 - val_loss: 0.9303 - val_accuracy: 0.8492 - val_jaccard_coef: 0.7316
Epoch 178/250
51/51 [==============================] - 61s 1s/step - loss: 0.8618 - accuracy: 0.9552 - jaccard_coef: 0.8996 - val_loss: 0.9327 - val_accuracy: 0.8463 - val_jaccard_coef: 0.7273
Epoch 179/250
51/51 [==============================] - 61s 1s/step - loss: 0.8629 - accuracy: 0.9517 - jaccard_coef: 0.8924 - val_loss: 0.9323 - val_accuracy: 0.8427 - val_jaccard_coef: 0.7205
Epoch 180/250
51/51 [==============================] - 61s 1s/step - loss: 0.8636 - accuracy: 0.9516 - jaccard_coef: 0.8922 - val_loss: 0.9304 - val_accuracy: 0.8476 - val_jaccard_coef: 0.7288
Epoch 181/250
51/51 [==============================] - 62s 1s/step - loss: 0.8622 - accuracy: 0.9547 - jaccard_coef: 0.8986 - val_loss: 0.9301 - val_accuracy: 0.8468 - val_jaccard_coef: 0.7277
Epoch 182/250
51/51 [==============================] - 61s 1s/step - loss: 0.8619 - accuracy: 0.9553 - jaccard_coef: 0.8997 - val_loss: 0.9284 - val_accuracy: 0.8508 - val_jaccard_coef: 0.7339
Epoch 183/250
51/51 [==============================] - 62s 1s/step - loss: 0.8664 - accuracy: 0.9442 - jaccard_coef: 0.8769 - val_loss: 0.9266 - val_accuracy: 0.8489 - val_jaccard_coef: 0.7268
Epoch 184/250
51/51 [==============================] - 61s 1s/step - loss: 0.8654 - accuracy: 0.9472 - jaccard_coef: 0.8816 - val_loss: 0.9231 - val_accuracy: 0.8504 - val_jaccard_coef: 0.7299
Epoch 185/250
51/51 [==============================] - 61s 1s/step - loss: 0.8662 - accuracy: 0.9451 - jaccard_coef: 0.8785 - val_loss: 0.9222 - val_accuracy: 0.8555 - val_jaccard_coef: 0.7379
Epoch 186/250
51/51 [==============================] - 61s 1s/step - loss: 0.8647 - accuracy: 0.9495 - jaccard_coef: 0.8873 - val_loss: 0.9252 - val_accuracy: 0.8554 - val_jaccard_coef: 0.7389
Epoch 187/250
51/51 [==============================] - 62s 1s/step - loss: 0.8652 - accuracy: 0.9485 - jaccard_coef: 0.8852 - val_loss: 0.9286 - val_accuracy: 0.8444 - val_jaccard_coef: 0.7236
Epoch 188/250
51/51 [==============================] - 61s 1s/step - loss: 0.8645 - accuracy: 0.9490 - jaccard_coef: 0.8865 - val_loss: 0.9279 - val_accuracy: 0.8506 - val_jaccard_coef: 0.7324
Epoch 189/250
51/51 [==============================] - 61s 1s/step - loss: 0.8644 - accuracy: 0.9487 - jaccard_coef: 0.8856 - val_loss: 0.9244 - val_accuracy: 0.8553 - val_jaccard_coef: 0.7364
Epoch 190/250
51/51 [==============================] - 61s 1s/step - loss: 0.8639 - accuracy: 0.9482 - jaccard_coef: 0.8850 - val_loss: 0.9213 - val_accuracy: 0.8618 - val_jaccard_coef: 0.7490
Epoch 191/250
51/51 [==============================] - 61s 1s/step - loss: 0.8634 - accuracy: 0.9504 - jaccard_coef: 0.8892 - val_loss: 0.9234 - val_accuracy: 0.8559 - val_jaccard_coef: 0.7407
Epoch 192/250
51/51 [==============================] - 61s 1s/step - loss: 0.8634 - accuracy: 0.9496 - jaccard_coef: 0.8878 - val_loss: 0.9178 - val_accuracy: 0.8627 - val_jaccard_coef: 0.7499
Epoch 193/250
51/51 [==============================] - 62s 1s/step - loss: 0.8626 - accuracy: 0.9523 - jaccard_coef: 0.8931 - val_loss: 0.9188 - val_accuracy: 0.8642 - val_jaccard_coef: 0.7533
Epoch 194/250
51/51 [==============================] - 62s 1s/step - loss: 0.8622 - accuracy: 0.9526 - jaccard_coef: 0.8943 - val_loss: 0.9189 - val_accuracy: 0.8592 - val_jaccard_coef: 0.7442
Epoch 195/250
51/51 [==============================] - 62s 1s/step - loss: 0.8630 - accuracy: 0.9508 - jaccard_coef: 0.8902 - val_loss: 0.9176 - val_accuracy: 0.8603 - val_jaccard_coef: 0.7462
Epoch 196/250
51/51 [==============================] - 62s 1s/step - loss: 0.8669 - accuracy: 0.9411 - jaccard_coef: 0.8703 - val_loss: 0.9240 - val_accuracy: 0.8550 - val_jaccard_coef: 0.7379
Epoch 197/250
51/51 [==============================] - 67s 1s/step - loss: 0.8670 - accuracy: 0.9400 - jaccard_coef: 0.8673 - val_loss: 0.9200 - val_accuracy: 0.8557 - val_jaccard_coef: 0.7388
Epoch 198/250
51/51 [==============================] - 65s 1s/step - loss: 0.8702 - accuracy: 0.9309 - jaccard_coef: 0.8501 - val_loss: 0.9218 - val_accuracy: 0.8524 - val_jaccard_coef: 0.7327
Epoch 199/250
51/51 [==============================] - 66s 1s/step - loss: 0.8645 - accuracy: 0.9459 - jaccard_coef: 0.8795 - val_loss: 0.9181 - val_accuracy: 0.8571 - val_jaccard_coef: 0.7391
Epoch 200/250
51/51 [==============================] - 68s 1s/step - loss: 0.8652 - accuracy: 0.9445 - jaccard_coef: 0.8765 - val_loss: 0.9176 - val_accuracy: 0.8609 - val_jaccard_coef: 0.7456
Epoch 201/250
51/51 [==============================] - 65s 1s/step - loss: 0.8633 - accuracy: 0.9496 - jaccard_coef: 0.8872 - val_loss: 0.9188 - val_accuracy: 0.8618 - val_jaccard_coef: 0.7478
Epoch 202/250
51/51 [==============================] - 63s 1s/step - loss: 0.8623 - accuracy: 0.9497 - jaccard_coef: 0.8881 - val_loss: 0.9147 - val_accuracy: 0.8663 - val_jaccard_coef: 0.7551
Epoch 203/250
51/51 [==============================] - 66s 1s/step - loss: 0.8604 - accuracy: 0.9533 - jaccard_coef: 0.8954 - val_loss: 0.9222 - val_accuracy: 0.8586 - val_jaccard_coef: 0.7451
Epoch 204/250
51/51 [==============================] - 72s 1s/step - loss: 0.8602 - accuracy: 0.9540 - jaccard_coef: 0.8968 - val_loss: 0.9196 - val_accuracy: 0.8614 - val_jaccard_coef: 0.7491
Epoch 205/250
51/51 [==============================] - 74s 1s/step - loss: 0.8623 - accuracy: 0.9496 - jaccard_coef: 0.8874 - val_loss: 0.9221 - val_accuracy: 0.8567 - val_jaccard_coef: 0.7406
Epoch 206/250
51/51 [==============================] - 75s 1s/step - loss: 0.8630 - accuracy: 0.9487 - jaccard_coef: 0.8854 - val_loss: 0.9204 - val_accuracy: 0.8592 - val_jaccard_coef: 0.7455
Epoch 207/250
51/51 [==============================] - 77s 2s/step - loss: 0.8622 - accuracy: 0.9492 - jaccard_coef: 0.8870 - val_loss: 0.9159 - val_accuracy: 0.8628 - val_jaccard_coef: 0.7501
Epoch 208/250
51/51 [==============================] - 72s 1s/step - loss: 0.8625 - accuracy: 0.9492 - jaccard_coef: 0.8869 - val_loss: 0.9162 - val_accuracy: 0.8643 - val_jaccard_coef: 0.7520
Epoch 209/250
51/51 [==============================] - 66s 1s/step - loss: 0.8613 - accuracy: 0.9508 - jaccard_coef: 0.8900 - val_loss: 0.9210 - val_accuracy: 0.8580 - val_jaccard_coef: 0.7423
Epoch 210/250
51/51 [==============================] - 65s 1s/step - loss: 0.8613 - accuracy: 0.9507 - jaccard_coef: 0.8902 - val_loss: 0.9174 - val_accuracy: 0.8516 - val_jaccard_coef: 0.7269
Epoch 211/250
51/51 [==============================] - 66s 1s/step - loss: 0.8808 - accuracy: 0.9105 - jaccard_coef: 0.8086 - val_loss: 0.9222 - val_accuracy: 0.8431 - val_jaccard_coef: 0.7138
Epoch 212/250
51/51 [==============================] - 64s 1s/step - loss: 0.8716 - accuracy: 0.9279 - jaccard_coef: 0.8423 - val_loss: 0.9159 - val_accuracy: 0.8579 - val_jaccard_coef: 0.7379
Epoch 213/250
51/51 [==============================] - 65s 1s/step - loss: 0.8672 - accuracy: 0.9368 - jaccard_coef: 0.8606 - val_loss: 0.9236 - val_accuracy: 0.8520 - val_jaccard_coef: 0.7318
Epoch 214/250
51/51 [==============================] - 63s 1s/step - loss: 0.8641 - accuracy: 0.9423 - jaccard_coef: 0.8718 - val_loss: 0.9193 - val_accuracy: 0.8573 - val_jaccard_coef: 0.7390
Epoch 215/250
51/51 [==============================] - 67s 1s/step - loss: 0.8629 - accuracy: 0.9468 - jaccard_coef: 0.8815 - val_loss: 0.9223 - val_accuracy: 0.8557 - val_jaccard_coef: 0.7384
Epoch 216/250
51/51 [==============================] - 68s 1s/step - loss: 0.8613 - accuracy: 0.9492 - jaccard_coef: 0.8863 - val_loss: 0.9241 - val_accuracy: 0.8536 - val_jaccard_coef: 0.7365
Epoch 217/250
51/51 [==============================] - 65s 1s/step - loss: 0.8598 - accuracy: 0.9520 - jaccard_coef: 0.8927 - val_loss: 0.9248 - val_accuracy: 0.8533 - val_jaccard_coef: 0.7368
Epoch 218/250
51/51 [==============================] - 66s 1s/step - loss: 0.8593 - accuracy: 0.9534 - jaccard_coef: 0.8955 - val_loss: 0.9278 - val_accuracy: 0.8484 - val_jaccard_coef: 0.7293
Epoch 219/250
51/51 [==============================] - 64s 1s/step - loss: 0.8588 - accuracy: 0.9538 - jaccard_coef: 0.8961 - val_loss: 0.9266 - val_accuracy: 0.8531 - val_jaccard_coef: 0.7367
Epoch 220/250
51/51 [==============================] - 63s 1s/step - loss: 0.8583 - accuracy: 0.9549 - jaccard_coef: 0.8990 - val_loss: 0.9230 - val_accuracy: 0.8590 - val_jaccard_coef: 0.7451
Epoch 221/250
51/51 [==============================] - 67s 1s/step - loss: 0.8584 - accuracy: 0.9545 - jaccard_coef: 0.8977 - val_loss: 0.9192 - val_accuracy: 0.8640 - val_jaccard_coef: 0.7531
Epoch 222/250
51/51 [==============================] - 65s 1s/step - loss: 0.8577 - accuracy: 0.9562 - jaccard_coef: 0.9014 - val_loss: 0.9208 - val_accuracy: 0.8621 - val_jaccard_coef: 0.7502
Epoch 223/250
51/51 [==============================] - 64s 1s/step - loss: 0.8573 - accuracy: 0.9571 - jaccard_coef: 0.9033 - val_loss: 0.9210 - val_accuracy: 0.8618 - val_jaccard_coef: 0.7497
Epoch 224/250
51/51 [==============================] - 66s 1s/step - loss: 0.8570 - accuracy: 0.9574 - jaccard_coef: 0.9042 - val_loss: 0.9215 - val_accuracy: 0.8590 - val_jaccard_coef: 0.7451
Epoch 225/250
51/51 [==============================] - 65s 1s/step - loss: 0.8574 - accuracy: 0.9567 - jaccard_coef: 0.9024 - val_loss: 0.9206 - val_accuracy: 0.8624 - val_jaccard_coef: 0.7504
Epoch 226/250
51/51 [==============================] - 68s 1s/step - loss: 0.8570 - accuracy: 0.9572 - jaccard_coef: 0.9038 - val_loss: 0.9228 - val_accuracy: 0.8588 - val_jaccard_coef: 0.7446
Epoch 227/250
51/51 [==============================] - 66s 1s/step - loss: 0.8575 - accuracy: 0.9568 - jaccard_coef: 0.9030 - val_loss: 0.9211 - val_accuracy: 0.8561 - val_jaccard_coef: 0.7404
Epoch 228/250
51/51 [==============================] - 64s 1s/step - loss: 0.8577 - accuracy: 0.9549 - jaccard_coef: 0.8987 - val_loss: 0.9201 - val_accuracy: 0.8571 - val_jaccard_coef: 0.7416
Epoch 229/250
51/51 [==============================] - 62s 1s/step - loss: 0.8573 - accuracy: 0.9562 - jaccard_coef: 0.9016 - val_loss: 0.9226 - val_accuracy: 0.8587 - val_jaccard_coef: 0.7445
Epoch 230/250
51/51 [==============================] - 61s 1s/step - loss: 0.8573 - accuracy: 0.9567 - jaccard_coef: 0.9025 - val_loss: 0.9245 - val_accuracy: 0.8546 - val_jaccard_coef: 0.7384
Epoch 231/250
51/51 [==============================] - 62s 1s/step - loss: 0.8567 - accuracy: 0.9578 - jaccard_coef: 0.9047 - val_loss: 0.9255 - val_accuracy: 0.8552 - val_jaccard_coef: 0.7400
Epoch 232/250
51/51 [==============================] - 61s 1s/step - loss: 0.8581 - accuracy: 0.9542 - jaccard_coef: 0.8973 - val_loss: 0.9237 - val_accuracy: 0.8567 - val_jaccard_coef: 0.7419
Epoch 233/250
51/51 [==============================] - 61s 1s/step - loss: 0.8583 - accuracy: 0.9540 - jaccard_coef: 0.8967 - val_loss: 0.9175 - val_accuracy: 0.8607 - val_jaccard_coef: 0.7465
Epoch 234/250
51/51 [==============================] - 61s 1s/step - loss: 0.8579 - accuracy: 0.9551 - jaccard_coef: 0.8991 - val_loss: 0.9199 - val_accuracy: 0.8598 - val_jaccard_coef: 0.7465
Epoch 235/250
51/51 [==============================] - 61s 1s/step - loss: 0.8566 - accuracy: 0.9573 - jaccard_coef: 0.9038 - val_loss: 0.9194 - val_accuracy: 0.8642 - val_jaccard_coef: 0.7537
Epoch 236/250
51/51 [==============================] - 61s 1s/step - loss: 0.8571 - accuracy: 0.9565 - jaccard_coef: 0.9021 - val_loss: 0.9376 - val_accuracy: 0.8363 - val_jaccard_coef: 0.7124
Epoch 237/250
51/51 [==============================] - 64s 1s/step - loss: 0.8774 - accuracy: 0.9143 - jaccard_coef: 0.8173 - val_loss: 0.9231 - val_accuracy: 0.8457 - val_jaccard_coef: 0.6918
Epoch 238/250
51/51 [==============================] - 68s 1s/step - loss: 0.8690 - accuracy: 0.9306 - jaccard_coef: 0.8465 - val_loss: 0.9243 - val_accuracy: 0.8458 - val_jaccard_coef: 0.7206
Epoch 239/250
51/51 [==============================] - 66s 1s/step - loss: 0.8694 - accuracy: 0.9310 - jaccard_coef: 0.8485 - val_loss: 0.9211 - val_accuracy: 0.8555 - val_jaccard_coef: 0.7369
Epoch 240/250
51/51 [==============================] - 65s 1s/step - loss: 0.8617 - accuracy: 0.9466 - jaccard_coef: 0.8810 - val_loss: 0.9219 - val_accuracy: 0.8567 - val_jaccard_coef: 0.7386
Epoch 241/250
51/51 [==============================] - 68s 1s/step - loss: 0.8592 - accuracy: 0.9514 - jaccard_coef: 0.8908 - val_loss: 0.9205 - val_accuracy: 0.8575 - val_jaccard_coef: 0.7403
Epoch 242/250
51/51 [==============================] - 68s 1s/step - loss: 0.8576 - accuracy: 0.9547 - jaccard_coef: 0.8981 - val_loss: 0.9213 - val_accuracy: 0.8608 - val_jaccard_coef: 0.7473
Epoch 243/250
51/51 [==============================] - 67s 1s/step - loss: 0.8571 - accuracy: 0.9560 - jaccard_coef: 0.9009 - val_loss: 0.9225 - val_accuracy: 0.8591 - val_jaccard_coef: 0.7448
Epoch 244/250
51/51 [==============================] - 68s 1s/step - loss: 0.8567 - accuracy: 0.9568 - jaccard_coef: 0.9029 - val_loss: 0.9198 - val_accuracy: 0.8637 - val_jaccard_coef: 0.7515
Epoch 245/250
51/51 [==============================] - 68s 1s/step - loss: 0.8565 - accuracy: 0.9574 - jaccard_coef: 0.9040 - val_loss: 0.9225 - val_accuracy: 0.8602 - val_jaccard_coef: 0.7469
Epoch 246/250
51/51 [==============================] - 68s 1s/step - loss: 0.8561 - accuracy: 0.9581 - jaccard_coef: 0.9056 - val_loss: 0.9206 - val_accuracy: 0.8627 - val_jaccard_coef: 0.7506
Epoch 247/250
51/51 [==============================] - 66s 1s/step - loss: 0.8560 - accuracy: 0.9582 - jaccard_coef: 0.9057 - val_loss: 0.9195 - val_accuracy: 0.8647 - val_jaccard_coef: 0.7536
Epoch 248/250
51/51 [==============================] - 67s 1s/step - loss: 0.8558 - accuracy: 0.9588 - jaccard_coef: 0.9070 - val_loss: 0.9219 - val_accuracy: 0.8620 - val_jaccard_coef: 0.7505
Epoch 249/250
51/51 [==============================] - 68s 1s/step - loss: 0.8557 - accuracy: 0.9585 - jaccard_coef: 0.9064 - val_loss: 0.9177 - val_accuracy: 0.8685 - val_jaccard_coef: 0.7600
Epoch 250/250
51/51 [==============================] - 69s 1s/step - loss: 0.8554 - accuracy: 0.9592 - jaccard_coef: 0.9080 - val_loss: 0.9173 - val_accuracy: 0.8702 - val_jaccard_coef: 0.7635

# ===== Cell Separator =====

# model_history = model.fit(X_train, y_train,
#                           batch_size=30,
#                           verbose=1,
#                           epochs=10,
#                           validation_data=(X_test, y_test),
#                           callbacks = [plot_loss],
#                           shuffle=False)

# ===== Cell Separator =====

import tensorflow as tf
from tensorflow.keras import backend as K

def jaccard_coef(y_true, y_pred):
    # Flatten using tf.reshape
    y_true_flatten = tf.reshape(y_true, [-1])
    y_pred_flatten = tf.reshape(y_pred, [-1])

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection

    return intersection / (union + K.epsilon())

# ===== Cell Separator =====

def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.reshape(y_true, (-1,))
    y_pred_flatten = K.reshape(y_pred, (-1,))
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection
    return intersection / (union + K.epsilon())

# ===== Cell Separator =====

print("Train X:", X_train.shape, "Train y:", y_train.shape)
print("Test X:", X_test.shape, "Test y:", y_test.shape)

# ===== Cell Separator =====

Train X: (803, 256, 256, 3) Train y: (803, 256, 256, 6)
Test X: (142, 256, 256, 3) Test y: (142, 256, 256, 6)

# ===== Cell Separator =====

import tensorflow as tf
from tensorflow.keras import backend as K

def jaccard_coef(y_true, y_pred, smooth=1e-6):
    # flatten both
    y_true_flat = K.batch_flatten(y_true)
    y_pred_flat = K.batch_flatten(y_pred)

    intersection = K.sum(y_true_flat * y_pred_flat, axis=1)
    union = K.sum(y_true_flat, axis=1) + K.sum(y_pred_flat, axis=1) - intersection

    return K.mean((intersection + smooth) / (union + smooth))

# ===== Cell Separator =====

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', jaccard_coef]
)

# ===== Cell Separator =====

{'loss': [1.0054471492767334,
  0.9796633124351501,
  0.958881139755249,
  0.9525033831596375,
  0.9491528272628784,
  0.9473537802696228,
  0.9450496435165405,
  0.9394502639770508,
  0.9335131645202637,
  0.9327805042266846,
  0.9282593727111816,
  0.9258340001106262,
  0.9233288764953613,
  0.922579288482666,
  0.9208921194076538,
  0.9189279675483704,
  0.9175374507904053,
  0.9173455238342285,
  0.9187695980072021,
  0.914971649646759,
  0.9133304357528687,
  0.9120344519615173,
  0.9111996293067932,
  0.9100791811943054,
  0.9088961482048035,
  0.908018171787262,
  0.9078607559204102,
  0.9073717594146729,
  0.9092968702316284,
  0.9071125984191895,
  0.9057648181915283,
  0.9054490327835083,
  0.9079452157020569,
  0.9033560156822205,
  0.9001685976982117,
  0.8975897431373596,
  0.9013440012931824,
  0.898925244808197,
  0.89717698097229,
  0.8960654139518738,
  0.897962212562561,
  0.9008207321166992,
  0.9126502275466919,
  0.9010730981826782,
  0.8968716263771057,
  0.8947017192840576,
  0.8953328132629395,
  0.8939167261123657,
  0.8920326828956604,
  0.889924943447113,
  0.8909927606582642,
  0.8899171948432922,
  0.8923168182373047,
  0.8984628319740295,
  0.8948283791542053,
  0.892440140247345,
  0.889890193939209,
  0.8929774761199951,
  0.8896452188491821,
  0.8873542547225952,
  0.8863158226013184,
  0.8872777819633484,
  0.8850592970848083,
  0.8845259547233582,
  0.8839020133018494,
  0.8832845091819763,
  0.8835548162460327,
  0.8827713131904602,
  0.8841987252235413,
  0.8829634785652161,
  0.8816883563995361,
  0.8797715902328491,
  0.8791841864585876,
  0.879044234752655,
  0.8786060810089111,
  0.8789165616035461,
  0.8789761662483215,
  0.8776217103004456,
  0.8783625364303589,
  0.876548707485199,
  0.8774924278259277,
  0.8767288327217102,
  0.8759900331497192,
  0.8762170672416687,
  0.8776564598083496,
  0.8767601251602173,
  0.8779947757720947,
  0.8803953528404236,
  0.878662109375,
  0.8764422535896301,
  0.8778431415557861,
  0.8765347599983215,
  0.876440167427063,
  0.8767759203910828,
  0.8765031695365906,
  0.8732818365097046,
  0.8734548091888428,
  0.8720104694366455,
  0.872070848941803,
  0.8710829615592957,
  0.8718094229698181,
  0.8716265559196472,
  0.8708228468894958,
  0.873691201210022,
  0.8718146085739136,
  0.8709912896156311,
  0.8705780506134033,
  0.8698139786720276,
  0.8696375489234924,
  0.8694755434989929,
  0.8692722916603088,
  0.8687194585800171,
  0.8688185214996338,
  0.8696132302284241,
  0.8697554469108582,
  0.8703850507736206,
  0.8688594698905945,
  0.868739902973175,
  0.8682060837745667,
  0.869432806968689,
  0.879589855670929,
  0.8918371200561523,
  0.8846868872642517,
  0.8775690793991089,
  0.8770608901977539,
  0.8748224377632141,
  0.8712517023086548,
  0.8704634308815002,
  0.8690031170845032,
  0.8684297204017639,
  0.8676679730415344,
  0.8673507571220398,
  0.8671883344650269,
  0.8671044111251831,
  0.8689481616020203,
  0.8672756552696228,
  0.8669164776802063,
  0.8664681315422058,
  0.8666505813598633,
  0.8663608431816101,
  0.8663000464439392,
  0.8663851022720337,
  0.8668212294578552,
  0.8659818172454834,
  0.867186427116394,
  0.865852415561676,
  0.8724386692047119,
  0.8698211908340454,
  0.8664204478263855,
  0.8652898073196411,
  0.8651028275489807,
  0.8650817275047302,
  0.8649041056632996,
  0.8646755814552307,
  0.8646931052207947,
  0.8651374578475952,
  0.8648969531059265,
  0.8649863004684448,
  0.8648005723953247,
  0.8646769523620605,
  0.8647332191467285,
  0.8650771379470825,
  0.8643181920051575,
  0.864262044429779,
  0.8648002743721008,
  0.8642091751098633,
  0.8634431958198547,
  0.8631857633590698,
  0.8632777333259583,
  0.8628655672073364,
  0.8631360530853271,
  0.8643139004707336,
  0.8636344075202942,
  0.8626446723937988,
  0.862783670425415,
  0.8622937202453613,
  0.8623293042182922,
  0.8617725968360901,
  0.8629074096679688,
  0.8635525107383728,
  0.8622024059295654,
  0.8618541359901428,
  0.8663744330406189,
  0.8654455542564392,
  0.8661983013153076,
  0.8647360801696777,
  0.8651822209358215,
  0.8644629120826721,
  0.8643926382064819,
  0.8639324903488159,
  0.8634012937545776,
  0.863376796245575,
  0.8625728487968445,
  0.8622257113456726,
  0.8629515171051025,
  0.8668909668922424,
  0.8670083284378052,
  0.8701984286308289,
  0.8644707798957825,
  0.8652492165565491,
  0.8632776737213135,
  0.8622637987136841,
  0.8604034781455994,
  0.8602193593978882,
  0.8622938990592957,
  0.8630416989326477,
  0.8621993660926819,
  0.8624891042709351,
  0.8613176345825195,
  0.8612660765647888,
  0.8807510733604431,
  0.8716488480567932,
  0.8672109246253967,
  0.8641150593757629,
  0.8628934621810913,
  0.8613405227661133,
  0.8597776293754578,
  0.8593345880508423,
  0.8587958216667175,
  0.8583031892776489,
  0.8583829402923584,
  0.8576632142066956,
  0.857299268245697,
  0.8570148944854736,
  0.8574100136756897,
  0.8570250868797302,
  0.8574806451797485,
  0.8576958775520325,
  0.8573386669158936,
  0.8573023676872253,
  0.8566795587539673,
  0.8580672144889832,
  0.8583183288574219,
  0.8578891754150391,
  0.8566471934318542,
  0.8570623397827148,
  0.877393901348114,
  0.8690492510795593,
  0.8694301247596741,
  0.8616669774055481,
  0.859157145023346,
  0.857593834400177,
  0.8570630550384521,
  0.8567031621932983,
  0.8564878106117249,
  0.8561226725578308,
  0.8559631109237671,
  0.8558450937271118,
  0.855665922164917,
  0.8554486632347107],
 'accuracy': [0.5146879553794861,
  0.6650426983833313,
  0.7314218878746033,
  0.7470802068710327,
  0.7525007128715515,
  0.7579704523086548,
  0.7657486796379089,
  0.7821036577224731,
  0.7988706827163696,
  0.8020409941673279,
  0.8143188953399658,
  0.8206518888473511,
  0.826708972454071,
  0.8288041353225708,
  0.8331780433654785,
  0.8370148539543152,
  0.8409594893455505,
  0.8417283296585083,
  0.8387565016746521,
  0.8477476239204407,
  0.8524503707885742,
  0.856070876121521,
  0.8584945201873779,
  0.8614211082458496,
  0.86380934715271,
  0.8650903105735779,
  0.8651967644691467,
  0.8651328086853027,
  0.860723614692688,
  0.8649426698684692,
  0.8714095950126648,
  0.8683189749717712,
  0.8638902306556702,
  0.8741214275360107,
  0.8793323040008545,
  0.8832386136054993,
  0.8777157068252563,
  0.8816249966621399,
  0.8840891718864441,
  0.8843773007392883,
  0.882289707660675,
  0.8726218342781067,
  0.8473138809204102,
  0.8765714764595032,
  0.8828899264335632,
  0.8871191740036011,
  0.8859359622001648,
  0.8895753026008606,
  0.8940524458885193,
  0.8979917168617249,
  0.8955248594284058,
  0.8977816104888916,
  0.8938656449317932,
  0.8796254992485046,
  0.8886392116546631,
  0.8916105031967163,
  0.8975855708122253,
  0.8891244530677795,
  0.8982643485069275,
  0.9021450877189636,
  0.9058113694190979,
  0.9020907282829285,
  0.9069854617118835,
  0.9065405130386353,
  0.9088067412376404,
  0.9101759195327759,
  0.909422755241394,
  0.9106608629226685,
  0.9088324308395386,
  0.9106140732765198,
  0.9144189953804016,
  0.9182735681533813,
  0.9191924333572388,
  0.9187514781951904,
  0.9201694130897522,
  0.9195584058761597,
  0.9185674786567688,
  0.9219830632209778,
  0.9208052754402161,
  0.9242954254150391,
  0.9220260977745056,
  0.9231330752372742,
  0.9249160289764404,
  0.9235952496528625,
  0.9199800491333008,
  0.9220336079597473,
  0.918877363204956,
  0.9141727089881897,
  0.9167883396148682,
  0.9232314825057983,
  0.9188193082809448,
  0.9224070906639099,
  0.9223372936248779,
  0.921583354473114,
  0.9233983159065247,
  0.9300352931022644,
  0.9297273755073547,
  0.9331223964691162,
  0.9336649775505066,
  0.935249924659729,
  0.9346749186515808,
  0.935091495513916,
  0.9356891512870789,
  0.9291075468063354,
  0.9332419633865356,
  0.9356223344802856,
  0.9368227124214172,
  0.938468337059021,
  0.938548743724823,
  0.9390295147895813,
  0.9398605823516846,
  0.9409577250480652,
  0.9407198429107666,
  0.9388715624809265,
  0.9389645457267761,
  0.9357489347457886,
  0.9402250647544861,
  0.9401057362556458,
  0.9420475959777832,
  0.9407991766929626,
  0.9156573414802551,
  0.8892539739608765,
  0.9048395156860352,
  0.9200674891471863,
  0.9229175448417664,
  0.9254860877990723,
  0.934868335723877,
  0.9359081387519836,
  0.9397361874580383,
  0.9411405324935913,
  0.942974328994751,
  0.9437158703804016,
  0.9440410733222961,
  0.9444167613983154,
  0.9390355944633484,
  0.9435799717903137,
  0.9442862868309021,
  0.9453439116477966,
  0.945197582244873,
  0.9457083344459534,
  0.9455693960189819,
  0.9452704191207886,
  0.9442066550254822,
  0.9464941620826721,
  0.9437301754951477,
  0.9463857412338257,
  0.9318782091140747,
  0.9377795457839966,
  0.9457562565803528,
  0.948104977607727,
  0.9486377239227295,
  0.9486159086227417,
  0.9492224454879761,
  0.9493095278739929,
  0.9494113326072693,
  0.9482982754707336,
  0.9488470554351807,
  0.949605405330658,
  0.9494089484214783,
  0.9493927955627441,
  0.9493584632873535,
  0.948443591594696,
  0.950291097164154,
  0.9503894448280334,
  0.9487266540527344,
  0.9496244192123413,
  0.9515678286552429,
  0.9526419043540955,
  0.9525787830352783,
  0.9532803893089294,
  0.9529979228973389,
  0.9500627517700195,
  0.9516955614089966,
  0.9535637497901917,
  0.9530702829360962,
  0.9539903402328491,
  0.9544153809547424,
  0.9552017450332642,
  0.9516527056694031,
  0.951632022857666,
  0.9547193646430969,
  0.9552762508392334,
  0.9442152976989746,
  0.9471628665924072,
  0.9450945258140564,
  0.9495010375976562,
  0.9484573006629944,
  0.949002206325531,
  0.9486508369445801,
  0.9481717348098755,
  0.9503833055496216,
  0.9496470093727112,
  0.9522839784622192,
  0.9525567293167114,
  0.9508181810379028,
  0.9410670399665833,
  0.9399588704109192,
  0.9309247732162476,
  0.9458610415458679,
  0.9444987773895264,
  0.9495608806610107,
  0.9497379660606384,
  0.9533155560493469,
  0.9539711475372314,
  0.9495700597763062,
  0.9486837387084961,
  0.9492061138153076,
  0.9492312669754028,
  0.950842022895813,
  0.9507394433021545,
  0.9105274081230164,
  0.9278931021690369,
  0.9368303418159485,
  0.9423364996910095,
  0.9467887282371521,
  0.9491838812828064,
  0.9520435333251953,
  0.9534476399421692,
  0.9537549614906311,
  0.9549480080604553,
  0.9545490145683289,
  0.956235408782959,
  0.9570604562759399,
  0.9574336409568787,
  0.9567445516586304,
  0.9572431445121765,
  0.9568073749542236,
  0.9548913240432739,
  0.9562397599220276,
  0.9566940665245056,
  0.9577537178993225,
  0.9542473554611206,
  0.954020082950592,
  0.9550800323486328,
  0.9572916626930237,
  0.9564512372016907,
  0.9143126606941223,
  0.9305965900421143,
  0.9309586882591248,
  0.9465988874435425,
  0.9514275193214417,
  0.954716145992279,
  0.9560388922691345,
  0.9568094611167908,
  0.9574465751647949,
  0.9580675363540649,
  0.9581963419914246,
  0.9588048458099365,
  0.9584925770759583,
  0.9591580629348755],
 'jaccard_coef': [0.24184966087341309,
  0.38124629855155945,
  0.47075170278549194,
  0.5030332803726196,
  0.5162820816040039,
  0.5250664353370667,
  0.5382585525512695,
  0.5728750228881836,
  0.6033477187156677,
  0.6048648953437805,
  0.6255519986152649,
  0.6369726061820984,
  0.6471794247627258,
  0.6506962776184082,
  0.6566964983940125,
  0.6654354929924011,
  0.6729718446731567,
  0.6725809574127197,
  0.6661686897277832,
  0.6842120289802551,
  0.690909743309021,
  0.6982618570327759,
  0.7013262510299683,
  0.7074519395828247,
  0.7124485969543457,
  0.7187031507492065,
  0.7200608849525452,
  0.7224397659301758,
  0.7136180400848389,
  0.7222518920898438,
  0.7317180037498474,
  0.7290425300598145,
  0.7189123034477234,
  0.7375737428665161,
  0.7486745119094849,
  0.7559487819671631,
  0.7433063983917236,
  0.7523208260536194,
  0.7557657361030579,
  0.7575534582138062,
  0.7525365948677063,
  0.7371373772621155,
  0.6915324926376343,
  0.7403791546821594,
  0.7543141841888428,
  0.7621414065361023,
  0.7610895037651062,
  0.7655516266822815,
  0.7755925059318542,
  0.7831676602363586,
  0.7781599164009094,
  0.7826051712036133,
  0.7737743854522705,
  0.7488767504692078,
  0.7644175887107849,
  0.7709464430809021,
  0.7807788252830505,
  0.7661647200584412,
  0.7825533747673035,
  0.7908827066421509,
  0.7984078526496887,
  0.7906796336174011,
  0.800713062286377,
  0.8006072640419006,
  0.803915798664093,
  0.8071276545524597,
  0.8058232069015503,
  0.8080571293830872,
  0.8049339056015015,
  0.8082882761955261,
  0.815191924571991,
  0.8232720494270325,
  0.8250061273574829,
  0.8245259523391724,
  0.8270369172096252,
  0.8262946009635925,
  0.8239544034004211,
  0.8308671712875366,
  0.8288565278053284,
  0.8353148698806763,
  0.8308613896369934,
  0.8327774405479431,
  0.8367156386375427,
  0.834235668182373,
  0.8274145722389221,
  0.8310560584068298,
  0.8260096311569214,
  0.8148698806762695,
  0.8206403255462646,
  0.8331186771392822,
  0.8251553773880005,
  0.8320556282997131,
  0.8316105008125305,
  0.8301142454147339,
  0.8335140943527222,
  0.8471473455429077,
  0.846594512462616,
  0.8529070019721985,
  0.8540074825286865,
  0.8579215407371521,
  0.8565515279769897,
  0.8574826717376709,
  0.8586891889572144,
  0.8452097773551941,
  0.8534390330314636,
  0.8587159514427185,
  0.8607921004295349,
  0.8644798398017883,
  0.8645767569541931,
  0.8658722639083862,
  0.8672642707824707,
  0.8697580695152283,
  0.869304358959198,
  0.8656789660453796,
  0.8656079769134521,
  0.8589805364608765,
  0.8679582476615906,
  0.8678486347198486,
  0.8719184994697571,
  0.8689948916435242,
  0.8191836476325989,
  0.7684324383735657,
  0.7974244952201843,
  0.8268411159515381,
  0.8332157731056213,
  0.8383703827857971,
  0.8569394946098328,
  0.858738899230957,
  0.8670198917388916,
  0.8698917627334595,
  0.873678982257843,
  0.8754480481147766,
  0.8762644529342651,
  0.8769442439079285,
  0.8657936453819275,
  0.8748612403869629,
  0.8767712712287903,
  0.8790732622146606,
  0.8785403966903687,
  0.8795939683914185,
  0.8792628645896912,
  0.8784764409065247,
  0.8764503598213196,
  0.881402850151062,
  0.8756839632987976,
  0.8810884356498718,
  0.852016806602478,
  0.8626808524131775,
  0.8793417811393738,
  0.8841255903244019,
  0.8854801058769226,
  0.8857961893081665,
  0.8869547247886658,
  0.8875066041946411,
  0.8875539302825928,
  0.885159969329834,
  0.8858572840690613,
  0.8879019618034363,
  0.8874622583389282,
  0.8873665928840637,
  0.8872953653335571,
  0.8853691816329956,
  0.8891361355781555,
  0.8894097805023193,
  0.8863075971603394,
  0.887612521648407,
  0.8918748497962952,
  0.8940073847770691,
  0.8939899206161499,
  0.8957680463790894,
  0.8950425982475281,
  0.8887492418289185,
  0.8920652270317078,
  0.8959301114082336,
  0.8950072526931763,
  0.8971099257469177,
  0.8979683518409729,
  0.899641752243042,
  0.8924149870872498,
  0.8922483325004578,
  0.8985543847084045,
  0.8996707201004028,
  0.8769024014472961,
  0.8816449046134949,
  0.8784563541412354,
  0.887303352355957,
  0.8852262496948242,
  0.8864978551864624,
  0.8856177926063538,
  0.8849656581878662,
  0.8891929388046265,
  0.8877899050712585,
  0.8930589556694031,
  0.894314706325531,
  0.890161395072937,
  0.8702681064605713,
  0.8673178553581238,
  0.8501051068305969,
  0.8795104622840881,
  0.8765381574630737,
  0.8872474431991577,
  0.8880818486213684,
  0.8953580856323242,
  0.8967640995979309,
  0.8874456882476807,
  0.8853539228439331,
  0.8869943022727966,
  0.8869494795799255,
  0.8900439739227295,
  0.8901537656784058,
  0.8086456060409546,
  0.842254102230072,
  0.860637903213501,
  0.8718294501304626,
  0.8815140128135681,
  0.8862779140472412,
  0.8926644921302795,
  0.8955352902412415,
  0.8961332440376282,
  0.8990448117256165,
  0.8977490663528442,
  0.9013651609420776,
  0.9032728672027588,
  0.9042026400566101,
  0.9024330377578735,
  0.9038241505622864,
  0.9030066728591919,
  0.8986796736717224,
  0.9016250967979431,
  0.9025143384933472,
  0.9046849012374878,
  0.8973405957221985,
  0.8966941237449646,
  0.8991036415100098,
  0.9038273692131042,
  0.9021459817886353,
  0.81728595495224,
  0.8464980125427246,
  0.8484973311424255,
  0.8809714913368225,
  0.8908392786979675,
  0.8981314897537231,
  0.9009059071540833,
  0.9029145240783691,
  0.9040262699127197,
  0.9055898189544678,
  0.9057332873344421,
  0.9070146083831787,
  0.9064351916313171,
  0.9079873561859131],
 'val_loss': [0.9909022450447083,
  0.9693900942802429,
  0.9676639437675476,
  0.9798979163169861,
  0.9728515148162842,
  0.9761452674865723,
  0.9754408597946167,
  0.961903989315033,
  0.9680547714233398,
  0.9500459432601929,
  0.9472044706344604,
  0.9447586536407471,
  0.9439400434494019,
  0.9409236907958984,
  0.9371718168258667,
  0.9349623322486877,
  0.9334343671798706,
  0.9372000098228455,
  0.934965968132019,
  0.9318436980247498,
  0.9316470623016357,
  0.9281539916992188,
  0.9310847520828247,
  0.9312993288040161,
  0.9321274161338806,
  0.9344373941421509,
  0.9280024766921997,
  0.934353232383728,
  0.9246285557746887,
  0.9217295050621033,
  0.9164140820503235,
  0.9205067753791809,
  0.9166871309280396,
  0.9181716442108154,
  0.9181752800941467,
  0.9200931787490845,
  0.9227574467658997,
  0.9153621792793274,
  0.9224545359611511,
  0.9329370856285095,
  0.9223278164863586,
  0.9256467223167419,
  0.921646237373352,
  0.9179826378822327,
  0.9180539846420288,
  0.913394033908844,
  0.9165765643119812,
  0.9125973582267761,
  0.9148851633071899,
  0.9156796932220459,
  0.9158506989479065,
  0.9155772924423218,
  0.9298463463783264,
  0.9142186045646667,
  0.9135326147079468,
  0.9134914875030518,
  0.9220278263092041,
  0.9193673729896545,
  0.9168217778205872,
  0.913898766040802,
  0.9107457399368286,
  0.9147582650184631,
  0.9122781753540039,
  0.9176391959190369,
  0.9151353240013123,
  0.9128786325454712,
  0.9141227602958679,
  0.9106537699699402,
  0.9109421968460083,
  0.9136594533920288,
  0.914232611656189,
  0.9153331518173218,
  0.9141833782196045,
  0.9184808731079102,
  0.9134884476661682,
  0.9151134490966797,
  0.9164444804191589,
  0.9235807657241821,
  0.9192555546760559,
  0.9164785742759705,
  0.916195273399353,
  0.920698881149292,
  0.925613522529602,
  0.9281224012374878,
  0.9182507395744324,
  0.9244462251663208,
  0.9276717305183411,
  0.9222289323806763,
  0.9263153076171875,
  0.929399311542511,
  0.9289299249649048,
  0.9200296401977539,
  0.9209261536598206,
  0.9202244877815247,
  0.9177818894386292,
  0.9185823202133179,
  0.9168661236763,
  0.9172163605690002,
  0.9214072227478027,
  0.9143972396850586,
  0.9174652099609375,
  0.9215371608734131,
  0.9173134565353394,
  0.9125099778175354,
  0.9161964654922485,
  0.9207823872566223,
  0.9212779998779297,
  0.9217848181724548,
  0.9214895367622375,
  0.9175883531570435,
  0.9202361702919006,
  0.9234954118728638,
  0.9214837551116943,
  0.9159750938415527,
  0.9248878359794617,
  0.9207384586334229,
  0.921845555305481,
  0.9200031161308289,
  0.9206600189208984,
  0.9297946095466614,
  0.9269164204597473,
  0.9199172258377075,
  0.9195955395698547,
  0.9217418432235718,
  0.9273990988731384,
  0.9167225956916809,
  0.9160017371177673,
  0.9188999533653259,
  0.9214181900024414,
  0.919955849647522,
  0.9182029962539673,
  0.9222404360771179,
  0.9199865460395813,
  0.9243813157081604,
  0.9213485717773438,
  0.9236860871315002,
  0.9210366010665894,
  0.9226968288421631,
  0.9239128232002258,
  0.9170980453491211,
  0.9199755787849426,
  0.9213913679122925,
  0.9190265536308289,
  0.9209818243980408,
  0.9173054099082947,
  0.9203898310661316,
  0.9161630272865295,
  0.9157867431640625,
  0.9179979562759399,
  0.9139090776443481,
  0.9168366193771362,
  0.9134481549263,
  0.9142099022865295,
  0.9165561199188232,
  0.9204472303390503,
  0.9234552383422852,
  0.926971435546875,
  0.9290164113044739,
  0.93079674243927,
  0.9316651225090027,
  0.9226359128952026,
  0.9303749799728394,
  0.9280628561973572,
  0.9284329414367676,
  0.9231511354446411,
  0.9254218339920044,
  0.9210205078125,
  0.9173847436904907,
  0.9193124771118164,
  0.9215058088302612,
  0.922610878944397,
  0.9202489852905273,
  0.9212099313735962,
  0.9259098768234253,
  0.9250045418739319,
  0.9302936792373657,
  0.9303200840950012,
  0.9327008128166199,
  0.9323050379753113,
  0.930437445640564,
  0.930127739906311,
  0.928377628326416,
  0.9266079068183899,
  0.9231142997741699,
  0.9221864938735962,
  0.9251649975776672,
  0.9285593032836914,
  0.927937388420105,
  0.9244258999824524,
  0.9213179349899292,
  0.9233631491661072,
  0.9178383946418762,
  0.9187886118888855,
  0.9188703894615173,
  0.9175958633422852,
  0.9239946007728577,
  0.9199743270874023,
  0.9217677712440491,
  0.9180707335472107,
  0.9175711870193481,
  0.9187912344932556,
  0.9147336483001709,
  0.9221909046173096,
  0.9196059703826904,
  0.9221447110176086,
  0.9203732013702393,
  0.9158730506896973,
  0.9162318110466003,
  0.9209555983543396,
  0.9173766374588013,
  0.9221680164337158,
  0.9159190058708191,
  0.9236001968383789,
  0.9193050861358643,
  0.9223321080207825,
  0.9240691661834717,
  0.9248262643814087,
  0.9278466701507568,
  0.9265884757041931,
  0.9230286478996277,
  0.9191758036613464,
  0.9208155870437622,
  0.9210048317909241,
  0.92149817943573,
  0.9206231236457825,
  0.9228177070617676,
  0.9211455583572388,
  0.9200595021247864,
  0.9225848317146301,
  0.924541175365448,
  0.9255033731460571,
  0.9237110018730164,
  0.9174681901931763,
  0.9198520183563232,
  0.9193844795227051,
  0.9376278519630432,
  0.9231453537940979,
  0.9243001937866211,
  0.9210599660873413,
  0.9219188690185547,
  0.9205191135406494,
  0.9212627410888672,
  0.9224865436553955,
  0.9197733998298645,
  0.9224681854248047,
  0.9206008911132812,
  0.9194570183753967,
  0.9218605160713196,
  0.9177298545837402,
  0.9173235893249512],
 'val_accuracy': [0.5880952477455139,
  0.7014328837394714,
  0.7062444686889648,
  0.6514596939086914,
  0.6843492984771729,
  0.6821573972702026,
  0.6937118172645569,
  0.7279915809631348,
  0.7275027632713318,
  0.762613832950592,
  0.7707163691520691,
  0.7780280113220215,
  0.7798935770988464,
  0.7847880125045776,
  0.7935929894447327,
  0.7970673441886902,
  0.8066609501838684,
  0.7953829765319824,
  0.8010601997375488,
  0.8055940270423889,
  0.8092117309570312,
  0.8173715472221375,
  0.8116085529327393,
  0.8081793785095215,
  0.8146533370018005,
  0.8044725060462952,
  0.8170990347862244,
  0.8029332756996155,
  0.8237373232841492,
  0.8393599987030029,
  0.8479429483413696,
  0.8367081880569458,
  0.8462131023406982,
  0.8388669490814209,
  0.8459370732307434,
  0.8376485109329224,
  0.8248943090438843,
  0.8407078981399536,
  0.830525815486908,
  0.8124809861183167,
  0.8305442929267883,
  0.823079526424408,
  0.8365185260772705,
  0.8384630680084229,
  0.8459122180938721,
  0.8519410490989685,
  0.8455227017402649,
  0.8536103963851929,
  0.851334273815155,
  0.8578054904937744,
  0.8586488366127014,
  0.8524167537689209,
  0.8219707608222961,
  0.8497743010520935,
  0.8547589778900146,
  0.852275013923645,
  0.8245187401771545,
  0.850694477558136,
  0.853365957736969,
  0.8566952347755432,
  0.8606425523757935,
  0.8552429676055908,
  0.8606009483337402,
  0.8545455932617188,
  0.8568772077560425,
  0.8646577596664429,
  0.859639048576355,
  0.8659152984619141,
  0.8634989857673645,
  0.8611546754837036,
  0.8625171184539795,
  0.8608325123786926,
  0.8600584268569946,
  0.855880856513977,
  0.8649457693099976,
  0.8598521947860718,
  0.8561686277389526,
  0.8448535799980164,
  0.8521630764007568,
  0.8592855930328369,
  0.8595249056816101,
  0.85533607006073,
  0.8457118272781372,
  0.8369781970977783,
  0.8502532243728638,
  0.8449212908744812,
  0.8346346020698547,
  0.8456764817237854,
  0.8380093574523926,
  0.8332825899124146,
  0.8356296420097351,
  0.8519071340560913,
  0.8533347845077515,
  0.8511268496513367,
  0.8560488224029541,
  0.8580858707427979,
  0.8570063710212708,
  0.8564079999923706,
  0.8529831767082214,
  0.8600208163261414,
  0.8558005690574646,
  0.8544055819511414,
  0.8622097969055176,
  0.8625675439834595,
  0.8588925004005432,
  0.8528503775596619,
  0.8520295023918152,
  0.8533587455749512,
  0.8561127185821533,
  0.8605411052703857,
  0.85728520154953,
  0.8521122336387634,
  0.8554556369781494,
  0.8632680177688599,
  0.8498516082763672,
  0.8546444773674011,
  0.8568622469902039,
  0.8574138283729553,
  0.8598947525024414,
  0.8461126685142517,
  0.8341388702392578,
  0.8390517830848694,
  0.8541638851165771,
  0.8490264415740967,
  0.842295229434967,
  0.8530518412590027,
  0.8627796173095703,
  0.8558708429336548,
  0.8532575964927673,
  0.8584839701652527,
  0.8606333136558533,
  0.8544509410858154,
  0.8563985824584961,
  0.8511825203895569,
  0.8484644293785095,
  0.849974513053894,
  0.8543874025344849,
  0.8527669906616211,
  0.8469895124435425,
  0.8584590554237366,
  0.8546716570854187,
  0.8520808815956116,
  0.8564754128456116,
  0.8534907102584839,
  0.8642330765724182,
  0.8559780716896057,
  0.8484410047531128,
  0.8624488115310669,
  0.8592494130134583,
  0.8666728734970093,
  0.8641990423202515,
  0.86857670545578,
  0.8676413893699646,
  0.8658556938171387,
  0.8601256608963013,
  0.8540806174278259,
  0.852889895439148,
  0.8461999893188477,
  0.8423696160316467,
  0.8467236161231995,
  0.8511708378791809,
  0.8469100594520569,
  0.8497356176376343,
  0.852292001247406,
  0.8558588027954102,
  0.8560048341751099,
  0.8620555996894836,
  0.865678608417511,
  0.8655177354812622,
  0.8607503175735474,
  0.8573509454727173,
  0.8607975244522095,
  0.8587631583213806,
  0.8564748764038086,
  0.8570756316184998,
  0.8479733467102051,
  0.849197268486023,
  0.8462979197502136,
  0.8427456021308899,
  0.8476319909095764,
  0.8468064069747925,
  0.8507887125015259,
  0.8489382266998291,
  0.8503561019897461,
  0.855498194694519,
  0.8554075956344604,
  0.8444015979766846,
  0.8505834937095642,
  0.8552609086036682,
  0.8618178963661194,
  0.8559257388114929,
  0.8627482652664185,
  0.8642160296440125,
  0.8591811656951904,
  0.8603052496910095,
  0.8550248742103577,
  0.8557178378105164,
  0.8523643612861633,
  0.8570981025695801,
  0.8608711957931519,
  0.8617966175079346,
  0.8663421273231506,
  0.8585899472236633,
  0.8613660335540771,
  0.8566821217536926,
  0.8591598868370056,
  0.8627955317497253,
  0.8643226027488708,
  0.8580074310302734,
  0.8516491055488586,
  0.8430701494216919,
  0.8579410910606384,
  0.8520001769065857,
  0.8573186993598938,
  0.8557344675064087,
  0.8535504341125488,
  0.8533002734184265,
  0.8483984470367432,
  0.8531313538551331,
  0.8590485453605652,
  0.8640041947364807,
  0.8620689511299133,
  0.8617987036705017,
  0.8590105175971985,
  0.8624477386474609,
  0.8587970733642578,
  0.8560636043548584,
  0.8570737242698669,
  0.8586764335632324,
  0.8545894622802734,
  0.85515296459198,
  0.8567234873771667,
  0.8607470989227295,
  0.8598408102989197,
  0.8642414808273315,
  0.8362559080123901,
  0.845686137676239,
  0.8458146452903748,
  0.8554673790931702,
  0.8567491173744202,
  0.8574638962745667,
  0.8608236312866211,
  0.8591361045837402,
  0.8636755347251892,
  0.8601884245872498,
  0.8626804351806641,
  0.8646573424339294,
  0.8619990944862366,
  0.8684985637664795,
  0.8702440857887268],
 'val_jaccard_coef': [0.3375292420387268,
  0.45424163341522217,
  0.4801432192325592,
  0.41318944096565247,
  0.45219558477401733,
  0.46391594409942627,
  0.49170196056365967,
  0.5309677124023438,
  0.5402384996414185,
  0.5794847011566162,
  0.5894458293914795,
  0.5977042317390442,
  0.6042600274085999,
  0.6039092540740967,
  0.6254255771636963,
  0.6288222074508667,
  0.6426128149032593,
  0.6252170205116272,
  0.6342586278915405,
  0.6363404393196106,
  0.6484282612800598,
  0.656945526599884,
  0.652898907661438,
  0.6456952095031738,
  0.6608102321624756,
  0.647682249546051,
  0.6613765358924866,
  0.6409258842468262,
  0.6635744571685791,
  0.6950565576553345,
  0.7066567540168762,
  0.6890577673912048,
  0.7066835761070251,
  0.7021287679672241,
  0.7115293741226196,
  0.7014099955558777,
  0.6773620843887329,
  0.6941671967506409,
  0.6881107687950134,
  0.6648709774017334,
  0.6869776248931885,
  0.6675902605056763,
  0.6928378343582153,
  0.6903845071792603,
  0.7138057351112366,
  0.721392810344696,
  0.7114332914352417,
  0.7214886546134949,
  0.7236807346343994,
  0.7318360805511475,
  0.7362269163131714,
  0.7228618860244751,
  0.6722227334976196,
  0.7141375541687012,
  0.7263143062591553,
  0.7184332013130188,
  0.6760390400886536,
  0.720114529132843,
  0.7269274592399597,
  0.7301485538482666,
  0.7364492416381836,
  0.7284860610961914,
  0.7365630269050598,
  0.7296092510223389,
  0.7317993640899658,
  0.742358386516571,
  0.7345784306526184,
  0.7455446720123291,
  0.7417722940444946,
  0.7379313707351685,
  0.7419212460517883,
  0.7396800518035889,
  0.7384045124053955,
  0.730487048625946,
  0.7461703419685364,
  0.7384312748908997,
  0.7342507243156433,
  0.7178088426589966,
  0.7284137010574341,
  0.7399190068244934,
  0.7390600442886353,
  0.7334657311439514,
  0.7205274105072021,
  0.7063047289848328,
  0.71222984790802,
  0.7181191444396973,
  0.7009536027908325,
  0.7116482257843018,
  0.7066795825958252,
  0.7031753659248352,
  0.7062194347381592,
  0.7288979291915894,
  0.7311387062072754,
  0.727968156337738,
  0.7363759875297546,
  0.7388492822647095,
  0.7381353974342346,
  0.7364330887794495,
  0.7339369654655457,
  0.7418053150177002,
  0.7367089986801147,
  0.7358238101005554,
  0.7462659478187561,
  0.7440483570098877,
  0.7398017644882202,
  0.7328561544418335,
  0.7310128211975098,
  0.7341846227645874,
  0.7384682297706604,
  0.7432032823562622,
  0.7392172813415527,
  0.7330400943756104,
  0.7376431226730347,
  0.7479687929153442,
  0.7293197512626648,
  0.7355278134346008,
  0.7399489879608154,
  0.739619791507721,
  0.7435609698295593,
  0.7231931686401367,
  0.6901404857635498,
  0.6939918994903564,
  0.727389931678772,
  0.7242082357406616,
  0.7154993414878845,
  0.7308645844459534,
  0.7476559281349182,
  0.7375448942184448,
  0.7342751026153564,
  0.7419739365577698,
  0.7466316819190979,
  0.7374228239059448,
  0.7387313842773438,
  0.731678307056427,
  0.7247824668884277,
  0.7300304174423218,
  0.7357823252677917,
  0.7339662313461304,
  0.7243472337722778,
  0.7423033118247986,
  0.73649001121521,
  0.7318103909492493,
  0.7390997409820557,
  0.7340462803840637,
  0.7508962750434875,
  0.7385804653167725,
  0.7181522846221924,
  0.7469699382781982,
  0.7422625422477722,
  0.7551859617233276,
  0.7516013383865356,
  0.7572357058525085,
  0.7566909790039062,
  0.75398188829422,
  0.7458333373069763,
  0.7361301183700562,
  0.7361487150192261,
  0.7230783104896545,
  0.720552921295166,
  0.7261415719985962,
  0.7326333522796631,
  0.7275784015655518,
  0.7312756776809692,
  0.7352137565612793,
  0.7387799620628357,
  0.7405292391777039,
  0.749814510345459,
  0.7540177702903748,
  0.7543210387229919,
  0.7472545504570007,
  0.7422964572906494,
  0.7468760013580322,
  0.7441584467887878,
  0.7417513728141785,
  0.7424778342247009,
  0.7297436594963074,
  0.7315840721130371,
  0.7273107767105103,
  0.7205016613006592,
  0.7287865281105042,
  0.7277130484580994,
  0.7338939309120178,
  0.7268442511558533,
  0.7298818230628967,
  0.7379317879676819,
  0.7389232516288757,
  0.7235655188560486,
  0.7323642373085022,
  0.7363882064819336,
  0.7490031719207764,
  0.7407400608062744,
  0.7499403357505798,
  0.7533146142959595,
  0.7442168593406677,
  0.7462074756622314,
  0.7379398345947266,
  0.7388397455215454,
  0.7326840162277222,
  0.7390537261962891,
  0.7456256151199341,
  0.7478440999984741,
  0.7551498413085938,
  0.7450637817382812,
  0.749126136302948,
  0.7405946254730225,
  0.7455118298530579,
  0.7500865459442139,
  0.7519738674163818,
  0.7423403263092041,
  0.7269341945648193,
  0.7138481140136719,
  0.7379072308540344,
  0.7318375110626221,
  0.7390235662460327,
  0.7384052872657776,
  0.7365313172340393,
  0.7368032336235046,
  0.7292777895927429,
  0.7367117404937744,
  0.7450634837150574,
  0.7531448602676392,
  0.7501780986785889,
  0.7497029900550842,
  0.7451316118240356,
  0.7503812313079834,
  0.7446346879005432,
  0.7403538823127747,
  0.7416106462478638,
  0.7445411086082458,
  0.7384177446365356,
  0.740014910697937,
  0.7418762445449829,
  0.7465322613716125,
  0.7465044856071472,
  0.7536892294883728,
  0.712436854839325,
  0.6917714476585388,
  0.72063148021698,
  0.7369354367256165,
  0.7386496663093567,
  0.740297257900238,
  0.7472563982009888,
  0.7448070645332336,
  0.7515227198600769,
  0.7469255924224854,
  0.7505624294281006,
  0.753638505935669,
  0.7504897117614746,
  0.7599719166755676,
  0.7634808421134949]}

# ===== Cell Separator =====

loss = history_a.history['loss']
val_loss = history_a.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training Vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ===== Cell Separator =====

jaccard_coef = history_a.history['jaccard_coef']
val_jaccard_coef = history_a.history['val_jaccard_coef']

epochs = range(1, len(jaccard_coef) + 1)
plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
plt.plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
plt.title("Training Vs Validation IoU")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ===== Cell Separator =====

array([[[[9.23821423e-03, 8.30290794e-01, 4.44641337e-02,
          6.69398531e-02, 4.88701053e-02, 1.96845765e-04],
         [9.52569710e-04, 8.83870006e-01, 3.69157605e-02,
          6.60684481e-02, 1.21559352e-02, 3.72072573e-05],
         [8.27447104e-04, 8.97062302e-01, 2.22855825e-02,
          5.87953031e-02, 2.10016742e-02, 2.76603969e-05],
         ...,
         [5.16361743e-03, 5.46638548e-01, 3.13623808e-02,
          3.93922776e-01, 2.27923598e-02, 1.20358971e-04],
         [5.37886191e-03, 4.92797732e-01, 3.42465304e-02,
          4.42538351e-01, 2.49551907e-02, 8.32877558e-05],
         [1.38290627e-02, 4.32128757e-01, 7.93712139e-02,
          4.27251846e-01, 4.66495194e-02, 7.69603706e-04]],

        [[1.09784643e-03, 8.81807923e-01, 4.18651886e-02,
          5.72131276e-02, 1.79768633e-02, 3.91042486e-05],
         [5.33370658e-05, 9.44513321e-01, 1.15992902e-02,
          3.86092663e-02, 5.22355689e-03, 1.28896772e-06],
         [5.02780713e-05, 9.47951972e-01, 7.87375960e-03,
          3.52396891e-02, 8.88288859e-03, 1.31209924e-06],
         ...,
         [5.46602765e-04, 5.51133692e-01, 1.10302242e-02,
          4.31919485e-01, 5.35972230e-03, 1.03056036e-05],
         [5.47771750e-04, 5.19931257e-01, 1.07422406e-02,
          4.62451071e-01, 6.32277597e-03, 4.98193504e-06],
         [2.52033118e-03, 4.80003655e-01, 3.23918387e-02,
          4.69182670e-01, 1.58096887e-02, 9.18209553e-05]],

        [[9.47599183e-04, 9.08504486e-01, 2.54976191e-02,
          5.15942350e-02, 1.34326080e-02, 2.34968466e-05],
         [4.43802965e-05, 9.59800899e-01, 8.43700860e-03,
          2.82284282e-02, 3.48868384e-03, 6.00665203e-07],
         [4.60319243e-05, 9.51932728e-01, 7.84804299e-03,
          3.35790217e-02, 6.59301458e-03, 1.16794286e-06],
         ...,
         [2.13495587e-04, 4.64117318e-01, 5.17127989e-03,
          5.27550757e-01, 2.94411159e-03, 3.01995124e-06],
         [1.89293933e-04, 4.30150360e-01, 3.73616442e-03,
          5.60613692e-01, 5.30834030e-03, 2.10874987e-06],
         [1.34919095e-03, 4.24336731e-01, 1.66712031e-02,
          5.41267514e-01, 1.63123906e-02, 6.29687420e-05]],

        ...,

        [[7.82035470e-01, 1.99492916e-01, 1.66433514e-03,
          8.41116533e-04, 1.59641001e-02, 2.04165235e-06],
         [8.56248438e-01, 1.38971671e-01, 9.50372254e-04,
          3.08975665e-04, 3.52013251e-03, 3.79313775e-07],
         [9.14304137e-01, 8.08024928e-02, 1.25185912e-03,
          1.86836493e-04, 3.45419720e-03, 3.62930336e-07],
         ...,
         [2.09102891e-05, 9.97635007e-01, 2.20019836e-03,
          1.25329825e-04, 1.78794453e-05, 6.46198089e-07],
         [5.18632587e-05, 9.94713128e-01, 4.99593234e-03,
          1.73083914e-04, 6.52781164e-05, 5.82110317e-07],
         [7.20392622e-04, 9.81097102e-01, 1.46557270e-02,
          2.07349611e-03, 1.43967965e-03, 1.35662858e-05]],

        [[6.72645986e-01, 3.10179234e-01, 2.66049686e-03,
          9.44910338e-04, 1.35665815e-02, 2.74869967e-06],
         [7.48111904e-01, 2.46332511e-01, 1.71237066e-03,
          3.79682926e-04, 3.46282497e-03, 6.70770021e-07],
         [8.70311320e-01, 1.24192052e-01, 1.64585700e-03,
          2.79067492e-04, 3.57103301e-03, 6.28445491e-07],
         ...,
         [3.44396249e-05, 9.97281551e-01, 2.54048384e-03,
          1.30370798e-04, 1.27865469e-05, 3.68267138e-07],
         [7.85951634e-05, 9.94178653e-01, 5.45924669e-03,
          2.37052038e-04, 4.60104166e-05, 4.31697629e-07],
         [1.22072711e-03, 9.85502601e-01, 1.03663402e-02,
          1.78924587e-03, 1.11272710e-03, 8.36874460e-06]],

        [[5.81708610e-01, 3.52473557e-01, 2.35085543e-02,
          6.37771469e-03, 3.58806886e-02, 5.09816891e-05],
         [7.08167255e-01, 2.69105732e-01, 6.38381531e-03,
          1.70387118e-03, 1.46311745e-02, 8.13849238e-06],
         [8.09117556e-01, 1.70214921e-01, 6.70284778e-03,
          1.02629641e-03, 1.29288230e-02, 9.55267751e-06],
         ...,
         [2.08747675e-04, 9.95168984e-01, 3.72971664e-03,
          7.27613689e-04, 1.62853597e-04, 2.13882299e-06],
         [4.86845151e-04, 9.91880119e-01, 5.97529672e-03,
          1.15795806e-03, 4.97408852e-04, 2.31681520e-06],
         [4.56822803e-03, 9.69338715e-01, 1.32997548e-02,
          6.47093402e-03, 6.28235610e-03, 3.99610908e-05]]],


       [[[8.73378992e-01, 6.38003647e-02, 8.18220247e-03,
          2.62801466e-03, 5.19839153e-02, 2.65352355e-05],
         [9.57228720e-01, 2.69572791e-02, 1.41790090e-03,
          2.76905339e-04, 1.41187524e-02, 3.83797868e-07],
         [9.83291924e-01, 1.01112388e-02, 4.57032729e-04,
          4.71973981e-05, 6.09258981e-03, 1.99717647e-08],
         ...,
         [9.13465321e-01, 4.30191495e-02, 2.20241933e-03,
          5.66792267e-04, 4.07456346e-02, 7.06185631e-07],
         [8.81503046e-01, 7.70565420e-02, 4.55927523e-03,
          9.25279630e-04, 3.59534584e-02, 2.46052491e-06],
         [7.34979928e-01, 1.35535762e-01, 2.08542887e-02,
          6.06654817e-03, 1.02492154e-01, 7.13085319e-05]],

        [[9.64462459e-01, 2.35869866e-02, 1.70430215e-03,
          1.72602173e-04, 1.00731449e-02, 4.92161917e-07],
         [9.92556810e-01, 5.59942238e-03, 4.28394531e-04,
          6.53924508e-06, 1.40872330e-03, 3.46947271e-09],
         [9.98384237e-01, 1.02532620e-03, 8.69465948e-05,
          4.83035251e-07, 5.02931711e-04, 5.15301586e-11],
         ...,
         [9.79057431e-01, 1.22110769e-02, 4.24998871e-04,
          2.49866935e-05, 8.28156993e-03, 5.25257438e-09],
         [9.53248084e-01, 3.38780582e-02, 2.48717773e-03,
          6.19621569e-05, 1.03246495e-02, 6.03014954e-08],
         [8.50667477e-01, 8.68484527e-02, 1.02808969e-02,
          1.37763296e-03, 5.08198142e-02, 5.74052365e-06]],

        [[9.84219372e-01, 7.90705532e-03, 1.66153698e-03,
          5.13724844e-05, 6.16056146e-03, 1.33449049e-07],
         [9.97896910e-01, 1.01707596e-03, 2.50278565e-04,
          1.78456253e-06, 8.33865779e-04, 5.64865654e-10],
         [9.99343455e-01, 1.98874914e-04, 6.85563136e-05,
          9.53776862e-08, 3.89123161e-04, 9.06789054e-12],
         ...,
         [9.83796775e-01, 4.58338344e-03, 2.69679440e-04,
          7.53585118e-06, 1.13425842e-02, 6.92115754e-10],
         [9.71922219e-01, 1.51274391e-02, 7.77043926e-04,
          2.33942792e-05, 1.21499300e-02, 7.22594917e-09],
         [8.78608346e-01, 6.13359399e-02, 6.71145692e-03,
          8.69483047e-04, 5.24724796e-02, 2.32601064e-06]],

        ...,

        [[9.84945416e-01, 1.45961465e-02, 3.49978691e-05,
          4.73106593e-06, 4.18647222e-04, 4.07785750e-10],
         [9.97567832e-01, 2.36448739e-03, 1.20904369e-05,
          1.34936968e-07, 5.54459075e-05, 4.36498147e-12],
         [9.99566972e-01, 3.59044556e-04, 2.55950454e-05,
          1.96846006e-08, 4.84068914e-05, 1.66915125e-12],
         ...,
         [1.00000000e+00, 9.93179672e-09, 1.70836869e-13,
          5.04479309e-19, 2.42036902e-09, 3.45283598e-32],
         [9.99999523e-01, 4.97768553e-07, 4.13845867e-12,
          5.42074749e-17, 9.70238823e-09, 2.15640165e-28],
         [9.99887109e-01, 1.00841557e-04, 2.64628124e-08,
          1.46190185e-10, 1.19820115e-05, 2.49571144e-18]],

        [[9.85890269e-01, 1.36747081e-02, 1.74896355e-04,
          5.30276520e-06, 2.54819461e-04, 4.05339140e-09],
         [9.93984640e-01, 5.93849551e-03, 2.38746543e-05,
          5.68474832e-07, 5.24857569e-05, 3.71844847e-11],
         [9.98713613e-01, 1.24934304e-03, 9.55559517e-06,
          4.08162890e-08, 2.74768281e-05, 1.86976780e-12],
         ...,
         [9.99999762e-01, 2.75805945e-07, 2.64643640e-13,
          8.28879476e-18, 2.73693024e-09, 1.06822870e-30],
         [9.99992847e-01, 7.19314585e-06, 2.97737009e-11,
          3.87117520e-15, 2.12687983e-08, 3.43295207e-25],
         [9.99583900e-01, 3.88811110e-04, 1.97743532e-07,
          1.97695815e-09, 2.70867822e-05, 2.62975672e-16]],

        [[9.29152071e-01, 6.34231046e-02, 3.55143636e-03,
          2.95294885e-04, 3.57542373e-03, 2.68980875e-06],
         [9.83742893e-01, 1.55749032e-02, 2.47914606e-04,
          8.98252347e-06, 4.25185426e-04, 9.21210574e-09],
         [9.91487980e-01, 8.05827975e-03, 1.80349365e-04,
          1.44951139e-06, 2.71952158e-04, 8.08429157e-10],
         ...,
         [9.99958992e-01, 4.05936335e-05, 2.61679900e-10,
          1.91630023e-13, 3.72595139e-07, 1.88903081e-22],
         [9.99597847e-01, 3.98025848e-04, 1.26503732e-08,
          4.83314326e-11, 4.19449543e-06, 2.30192308e-18],
         [9.94017005e-01, 4.93941596e-03, 1.77551374e-05,
          1.01976832e-06, 1.02477404e-03, 3.08125608e-11]]],


       [[[3.48094702e-02, 6.26566052e-01, 2.66073048e-01,
          3.01586483e-02, 4.21050638e-02, 2.87739880e-04],
         [5.12065599e-03, 5.31143844e-01, 4.45748985e-01,
          1.04834726e-02, 7.47769512e-03, 2.52534373e-05],
         [2.89362622e-03, 4.58488524e-01, 5.24687111e-01,
          1.13651436e-02, 2.52022664e-03, 4.53414978e-05],
         ...,
         [1.88912894e-03, 4.35456127e-01, 4.19136994e-02,
          4.90079194e-01, 3.05854511e-02, 7.64814540e-05],
         [2.27671792e-03, 5.85032821e-01, 3.22800912e-02,
          3.52491379e-01, 2.78510600e-02, 6.78990400e-05],
         [8.52987450e-03, 5.28804302e-01, 7.51717612e-02,
          3.36381495e-01, 5.04360832e-02, 6.76391937e-04]],

        [[6.44297805e-03, 4.92864162e-01, 4.80089843e-01,
          1.05914827e-02, 9.98558849e-03, 2.59742410e-05],
         [4.71658510e-04, 3.85186940e-01, 6.10782325e-01,
          2.30213883e-03, 1.25607662e-03, 7.89132855e-07],
         [2.27164972e-04, 3.26174766e-01, 6.71723843e-01,
          1.56997005e-03, 3.03417095e-04, 8.98081623e-07],
         ...,
         [1.26904415e-04, 4.82257873e-01, 1.08639011e-02,
          4.98149246e-01, 8.59334413e-03, 8.71526390e-06],
         [1.63223784e-04, 6.32538915e-01, 1.00994678e-02,
          3.49097669e-01, 8.09357502e-03, 7.05320781e-06],
         [1.61149690e-03, 6.16863847e-01, 3.76103707e-02,
          3.21135223e-01, 2.26217899e-02, 1.57201575e-04]],

        [[4.55423817e-03, 3.28346938e-01, 6.49380445e-01,
          8.00709333e-03, 9.68800113e-03, 2.33900446e-05],
         [2.03050906e-04, 2.18459308e-01, 7.79493630e-01,
          1.15027849e-03, 6.93173613e-04, 5.07156983e-07],
         [8.66344490e-05, 1.51345551e-01, 8.47809792e-01,
          5.93181234e-04, 1.64111465e-04, 7.15516080e-07],
         ...,
         [1.29310152e-04, 5.49911559e-01, 7.07892468e-03,
          4.37149405e-01, 5.71715366e-03, 1.36271092e-05],
         [1.26454266e-04, 6.68031752e-01, 6.33447943e-03,
          3.16994458e-01, 8.50583240e-03, 7.03339219e-06],
         [1.23566121e-03, 6.44356012e-01, 2.93193422e-02,
          3.02979112e-01, 2.19807681e-02, 1.29125416e-04]],

        ...,

        [[3.77276610e-03, 6.57503784e-01, 3.14451605e-01,
          9.77873802e-03, 1.44471880e-02, 4.59032563e-05],
         [4.58849157e-04, 6.80039823e-01, 3.14096242e-01,
          2.06807791e-03, 3.33460653e-03, 2.32815319e-06],
         [2.68926902e-04, 7.34214485e-01, 2.62327999e-01,
          2.07947800e-03, 1.10510236e-03, 4.06591380e-06],
         ...,
         [2.47098378e-05, 2.28657983e-02, 9.75408673e-01,
          1.25633494e-04, 1.57520268e-03, 2.21907008e-08],
         [2.59597728e-05, 2.50444356e-02, 9.73015070e-01,
          1.23091435e-04, 1.79141248e-03, 2.08671533e-08],
         [7.17452553e-04, 5.34947962e-02, 9.34126556e-01,
          1.20630220e-03, 1.04527948e-02, 2.04082562e-06]],

        [[6.99365465e-03, 7.22191215e-01, 2.50864238e-01,
          1.07618691e-02, 9.08860471e-03, 1.00411940e-04],
         [1.18087512e-03, 7.86602020e-01, 2.07507715e-01,
          2.39779172e-03, 2.30687973e-03, 4.64611730e-06],
         [6.81060483e-04, 7.94791460e-01, 2.01207802e-01,
          2.33217888e-03, 9.81485588e-04, 6.09113113e-06],
         ...,
         [4.79051778e-05, 2.51552276e-02, 9.73260820e-01,
          1.48634281e-04, 1.38736086e-03, 3.03159524e-08],
         [6.86236381e-05, 2.87544783e-02, 9.69200552e-01,
          2.26889213e-04, 1.74949504e-03, 5.08999918e-08],
         [1.56016322e-03, 6.89932629e-02, 9.16742980e-01,
          1.64286781e-03, 1.10578043e-02, 2.96285407e-06]],

        [[2.84378417e-02, 6.46854043e-01, 2.70085543e-01,
          2.70016640e-02, 2.69560385e-02, 6.64926542e-04],
         [8.36398453e-03, 7.05092609e-01, 2.67574668e-01,
          1.00873020e-02, 8.79601482e-03, 8.54644895e-05],
         [5.28286397e-03, 6.93914294e-01, 2.85410374e-01,
          1.07685970e-02, 4.50749090e-03, 1.16280826e-04],
         ...,
         [1.27043179e-03, 6.67678714e-02, 9.25566077e-01,
          1.46039680e-03, 4.93227271e-03, 3.00316651e-06],
         [1.78379437e-03, 6.17022775e-02, 9.28216100e-01,
          1.67011411e-03, 6.62417198e-03, 3.58577063e-06],
         [1.26932943e-02, 1.14125766e-01, 8.26750576e-01,
          9.27258097e-03, 3.70265581e-02, 1.31306442e-04]]],


       ...,


       [[[6.39555827e-02, 8.04956734e-01, 9.70892087e-02,
          2.05317661e-02, 1.30871236e-02, 3.79610894e-04],
         [2.42236499e-02, 8.33184719e-01, 1.26010761e-01,
          1.31250974e-02, 3.34343174e-03, 1.12309113e-04],
         [3.40412706e-02, 7.89843440e-01, 1.53464645e-01,
          1.85965989e-02, 3.86221334e-03, 1.91818210e-04],
         ...,
         [5.84361330e-03, 9.87674952e-01, 3.23323486e-03,
          1.46496808e-03, 1.77507440e-03, 8.13978932e-06],
         [5.50522190e-03, 9.87629831e-01, 3.11993598e-03,
          1.35431986e-03, 2.38664192e-03, 4.06140953e-06],
         [1.32762380e-02, 9.50924397e-01, 2.01299284e-02,
          7.24654831e-03, 8.35053530e-03, 7.23261546e-05]],

        [[2.21293718e-02, 8.08396161e-01, 1.58452213e-01,
          8.28053895e-03, 2.65264395e-03, 8.90147785e-05],
         [9.04779043e-03, 8.51508021e-01, 1.35428071e-01,
          3.25810211e-03, 7.45387340e-04, 1.26819568e-05],
         [1.72334835e-02, 8.14170897e-01, 1.61802709e-01,
          5.62901096e-03, 1.11096213e-03, 5.28942910e-05],
         ...,
         [3.28077003e-04, 9.95143533e-01, 4.23999084e-03,
          2.31300350e-04, 5.43479618e-05, 2.77968343e-06],
         [3.47735739e-04, 9.95389342e-01, 3.89427366e-03,
          2.42399285e-04, 1.25049963e-04, 1.17217405e-06],
         [2.13904469e-03, 9.90182936e-01, 4.45569959e-03,
          1.51001045e-03, 1.70631427e-03, 5.99673922e-06]],

        [[4.13208231e-02, 8.22785974e-01, 1.25573426e-01,
          8.13035853e-03, 2.05971324e-03, 1.29801381e-04],
         [2.12351922e-02, 8.60547781e-01, 1.13714777e-01,
          3.76638933e-03, 7.00973673e-04, 3.48834255e-05],
         [4.43963408e-02, 8.18591654e-01, 1.27177864e-01,
          7.80049199e-03, 1.80626020e-03, 2.27476106e-04],
         ...,
         [1.25937964e-04, 9.97101128e-01, 2.66027427e-03,
          9.92207279e-05, 1.24230637e-05, 9.48571142e-07],
         [1.49401647e-04, 9.97076511e-01, 2.61680130e-03,
          1.16061223e-04, 4.09113927e-05, 3.69270992e-07],
         [1.10474322e-03, 9.93604600e-01, 3.70276021e-03,
          8.65678594e-04, 7.18866824e-04, 3.34902370e-06]],

        ...,

        [[3.44863646e-02, 9.57982719e-01, 2.33658892e-03,
          1.30325952e-03, 3.88020161e-03, 1.09030034e-05],
         [2.30993852e-02, 9.75034714e-01, 9.47344641e-04,
          3.76367068e-04, 5.39196422e-04, 2.99660292e-06],
         [3.50598507e-02, 9.62988257e-01, 1.27079966e-03,
          3.44794244e-04, 3.26519308e-04, 9.74684917e-06],
         ...,
         [9.99536514e-01, 4.40476637e-04, 3.63003409e-07,
          1.24821353e-09, 2.26295542e-05, 4.55367824e-16],
         [9.97649968e-01, 2.32077157e-03, 1.35597952e-06,
          7.05238223e-09, 2.78968510e-05, 1.13869214e-14],
         [9.82548237e-01, 1.61734391e-02, 6.16361212e-05,
          6.19914317e-06, 1.21047953e-03, 4.11356116e-10]],

        [[3.59625816e-02, 9.53938842e-01, 4.92558256e-03,
          2.09831586e-03, 3.06060817e-03, 1.41038208e-05],
         [2.52309125e-02, 9.73407090e-01, 5.82596113e-04,
          3.41273495e-04, 4.36330185e-04, 1.80179075e-06],
         [2.34655030e-02, 9.75510955e-01, 5.86988463e-04,
          2.60546629e-04, 1.73835637e-04, 2.20241623e-06],
         ...,
         [9.97791052e-01, 2.20270734e-03, 2.98456172e-07,
          2.17522556e-09, 5.79189918e-06, 9.74093713e-16],
         [9.91736829e-01, 8.24142806e-03, 2.09751715e-06,
          3.21168265e-08, 1.95776647e-05, 1.93299654e-13],
         [9.60209906e-01, 3.85550372e-02, 1.26328465e-04,
          1.45965232e-05, 1.09407492e-03, 2.61703881e-09]],

        [[7.13541061e-02, 8.94126117e-01, 1.43136177e-02,
          7.96236843e-03, 1.21761616e-02, 6.77170538e-05],
         [3.92060094e-02, 9.55249786e-01, 1.32394047e-03,
          1.35985750e-03, 2.85657286e-03, 3.80008123e-06],
         [2.72251293e-02, 9.70254183e-01, 7.58763927e-04,
          6.85576699e-04, 1.07360154e-03, 2.71878184e-06],
         ...,
         [9.84636545e-01, 1.52818682e-02, 2.60642423e-06,
          1.18878546e-07, 7.88262259e-05, 1.00008272e-12],
         [9.59544957e-01, 4.01133783e-02, 2.50910671e-05,
          2.37162567e-06, 3.14229255e-04, 1.74490547e-10],
         [8.86579990e-01, 1.02454476e-01, 8.16456217e-04,
          3.51288094e-04, 9.79734864e-03, 5.08055280e-07]]],


       [[[8.90029967e-01, 6.24040328e-02, 1.55433351e-02,
          2.55354261e-03, 2.94149034e-02, 5.42618145e-05],
         [9.73183930e-01, 1.90999471e-02, 1.67853315e-03,
          1.80998803e-04, 5.85612888e-03, 4.30412740e-07],
         [9.91768599e-01, 5.66101680e-03, 1.71629305e-04,
          1.31564693e-05, 2.38561770e-03, 3.13149995e-09],
         ...,
         [1.58690184e-01, 7.71081805e-01, 8.65764916e-03,
          2.20732410e-02, 3.92781459e-02, 2.18952540e-04],
         [1.77704066e-01, 7.58181334e-01, 1.23993400e-02,
          1.45457070e-02, 3.70075479e-02, 1.61956472e-04],
         [1.98582381e-01, 6.49120271e-01, 4.19545397e-02,
          5.00475354e-02, 5.89161254e-02, 1.37915742e-03]],

        [[9.79818702e-01, 1.45687684e-02, 1.81815028e-03,
          1.23191552e-04, 3.67058301e-03, 6.36576431e-07],
         [9.97243881e-01, 2.18708836e-03, 2.63131660e-04,
          1.22822394e-06, 3.04635789e-04, 8.06779199e-10],
         [9.99531269e-01, 3.48600588e-04, 1.78279934e-05,
          3.57096113e-08, 1.02197722e-04, 1.37572831e-12],
         ...,
         [1.28926098e-01, 8.50833714e-01, 3.09337978e-03,
          5.63411042e-03, 1.14851035e-02, 2.75381972e-05],
         [9.64746177e-02, 8.83643508e-01, 3.92169552e-03,
          4.28738911e-03, 1.16481856e-02, 2.46458185e-05],
         [1.65634155e-01, 7.54934847e-01, 1.50509812e-02,
          2.54830178e-02, 3.86028774e-02, 2.94094702e-04]],

        [[9.94539797e-01, 2.89899972e-03, 9.85717983e-04,
          1.28344536e-05, 1.56257558e-03, 3.58872967e-08],
         [9.99549568e-01, 2.46096897e-04, 9.53167546e-05,
          1.65947625e-07, 1.08778244e-04, 4.71987519e-11],
         [9.99907255e-01, 3.63988183e-05, 1.23033078e-05,
          4.00163636e-09, 4.41506963e-05, 1.36362967e-13],
         ...,
         [1.73234284e-01, 8.11201751e-01, 2.71381065e-03,
          4.49063024e-03, 8.32938589e-03, 3.01938089e-05],
         [1.35098115e-01, 8.52601588e-01, 2.24201172e-03,
          3.24680493e-03, 6.79682568e-03, 1.46642587e-05],
         [1.93574667e-01, 7.52211988e-01, 8.91128741e-03,
          1.98034626e-02, 2.53080837e-02, 1.90500912e-04]],

        ...,

        [[9.86359298e-01, 1.31737050e-02, 4.39477772e-05,
          6.54240284e-06, 4.16442053e-04, 8.81794859e-10],
         [9.98353124e-01, 1.60102430e-03, 1.00200195e-05,
          8.58843805e-08, 3.58384350e-05, 3.53774692e-12],
         [9.99770343e-01, 1.92462816e-04, 1.35028713e-05,
          6.87671475e-09, 2.37716849e-05, 4.15398754e-13],
         ...,
         [1.00000000e+00, 2.71738951e-08, 7.96861763e-13,
          4.80019031e-18, 8.74807693e-09, 1.51572483e-30],
         [9.99998927e-01, 1.05956769e-06, 1.54531735e-11,
          3.67458271e-16, 2.78466779e-08, 5.02943432e-27],
         [9.99827504e-01, 1.51438508e-04, 5.89897837e-08,
          4.13386214e-10, 2.10098115e-05, 1.50843664e-17]],

        [[9.88462389e-01, 1.11333495e-02, 2.01110073e-04,
          5.76085449e-06, 1.97391113e-04, 7.36209405e-09],
         [9.95367050e-01, 4.58884472e-03, 1.74609795e-05,
          3.70216583e-07, 2.63956845e-05, 2.78380808e-11],
         [9.99228597e-01, 7.56174501e-04, 4.80589779e-06,
          1.60927360e-08, 1.04562623e-05, 5.59718179e-13],
         ...,
         [9.99999166e-01, 8.36302092e-07, 1.92851182e-12,
          1.58447545e-16, 1.73163723e-08, 1.41927146e-28],
         [9.99982357e-01, 1.74959459e-05, 1.21998828e-10,
          3.24324003e-14, 8.86397089e-08, 9.47540986e-24],
         [9.99222159e-01, 7.20301876e-04, 4.68325482e-07,
          6.63088429e-09, 5.70878219e-05, 1.89430342e-15]],

        [[9.23861623e-01, 6.84911534e-02, 3.59621015e-03,
          3.57029377e-04, 3.69034475e-03, 3.74748129e-06],
         [9.86074209e-01, 1.34947030e-02, 1.91692714e-04,
          5.57740350e-06, 2.33848332e-04, 6.39744169e-09],
         [9.93863940e-01, 5.91240358e-03, 9.70561377e-05,
          7.52091012e-07, 1.25885272e-04, 3.14826470e-10],
         ...,
         [9.99902844e-01, 9.52620976e-05, 2.00752126e-09,
          2.42224079e-12, 1.87974274e-06, 1.27702986e-20],
         [9.99186099e-01, 7.98376219e-04, 5.11236564e-08,
          2.95345109e-10, 1.54668360e-05, 3.82472003e-17],
         [9.90619183e-01, 7.37333065e-03, 3.57087702e-05,
          2.29526790e-06, 1.96956750e-03, 1.01058398e-10]]],


       [[[8.38348150e-01, 7.82611892e-02, 1.71757564e-02,
          4.56510158e-03, 6.15552701e-02, 9.45319407e-05],
         [9.46253896e-01, 3.43298577e-02, 3.95673094e-03,
          5.77775063e-04, 1.48790190e-02, 2.67624341e-06],
         [9.76924539e-01, 1.46385031e-02, 8.26634176e-04,
          9.03635009e-05, 7.51983561e-03, 8.52093933e-08],
         ...,
         [9.95239735e-01, 3.88402888e-03, 2.93058183e-05,
          1.82738086e-06, 8.45044851e-04, 5.11346208e-11],
         [9.86358345e-01, 1.25424527e-02, 1.36028408e-04,
          6.85399300e-06, 9.56249656e-04, 7.44361406e-10],
         [9.14997160e-01, 6.09052330e-02, 3.12910439e-03,
          4.54138324e-04, 2.05138829e-02, 5.50278514e-07]],

        [[9.53680515e-01, 2.82969605e-02, 3.46705574e-03,
          3.99988348e-04, 1.41530689e-02, 2.42578380e-06],
         [9.89000261e-01, 8.15478712e-03, 1.06953946e-03,
          1.70867825e-05, 1.75830710e-03, 2.77229653e-08],
         [9.97262597e-01, 1.90518203e-03, 1.50041000e-04,
          1.09580390e-06, 6.81137433e-04, 2.56021315e-10],
         ...,
         [9.99628186e-01, 3.50004586e-04, 1.54725478e-06,
          4.09177447e-09, 2.02539013e-05, 8.39245242e-15],
         [9.98044729e-01, 1.88610714e-03, 1.38637797e-05,
          2.65443045e-08, 5.52478741e-05, 2.97134008e-13],
         [9.75685477e-01, 2.03396305e-02, 4.21949924e-04,
          1.95511875e-05, 3.53338150e-03, 2.84675394e-09]],

        [[9.75597978e-01, 1.11980513e-02, 2.75545730e-03,
          1.28702188e-04, 1.03192572e-02, 5.81040752e-07],
         [9.96319056e-01, 1.79653661e-03, 5.52937272e-04,
          5.12173619e-06, 1.32636132e-03, 3.89857568e-09],
         [9.98668313e-01, 4.63738514e-04, 1.36175266e-04,
          3.26332270e-07, 7.31430191e-04, 5.66304607e-11],
         ...,
         [9.99900103e-01, 4.53285174e-05, 5.64724189e-07,
          5.23822596e-10, 5.39983048e-05, 1.86874976e-16],
         [9.99472678e-01, 3.99449258e-04, 2.17694037e-06,
          3.60833741e-09, 1.25749779e-04, 4.77156656e-15],
         [9.87855613e-01, 7.49310199e-03, 1.72934291e-04,
          7.10559652e-06, 4.47115768e-03, 4.22261615e-10]],

        ...,

        [[9.27342415e-01, 7.02735558e-02, 4.41418175e-04,
          1.20031385e-04, 1.82239804e-03, 1.16513107e-07],
         [9.70918059e-01, 2.84516290e-02, 2.18295143e-04,
          1.32766145e-05, 3.98687029e-04, 7.17197368e-09],
         [9.88803566e-01, 1.02396822e-02, 4.46375838e-04,
          8.04963565e-06, 5.02263079e-04, 1.11349019e-08],
         ...,
         [9.99491334e-01, 2.47273128e-04, 2.42156966e-06,
          7.19225746e-09, 2.59010558e-04, 8.33076724e-15],
         [9.98550594e-01, 1.15245208e-03, 5.67114830e-06,
          3.47773934e-08, 2.91146891e-04, 1.37352843e-13],
         [9.83381927e-01, 1.19160973e-02, 2.24992749e-04,
          1.99227161e-05, 4.45697922e-03, 2.95246139e-09]],

        [[9.39601481e-01, 5.79606928e-02, 1.13547558e-03,
          1.07443368e-04, 1.19450153e-03, 4.49066761e-07],
         [9.52959299e-01, 4.63940464e-02, 2.86986586e-04,
          3.03691559e-05, 3.29339062e-04, 2.68816116e-08],
         [9.79199946e-01, 2.03202832e-02, 2.10777420e-04,
          7.93804429e-06, 2.61024834e-04, 7.97844013e-09],
         ...,
         [9.98495102e-01, 1.21319515e-03, 4.88528713e-06,
          4.51743674e-08, 2.86699709e-04, 1.53799418e-13],
         [9.95762825e-01, 3.80512932e-03, 1.52007588e-05,
          2.67500013e-07, 4.16563504e-04, 5.49391565e-12],
         [9.70130861e-01, 2.33045612e-02, 5.71282406e-04,
          6.56515331e-05, 5.92758786e-03, 2.92266780e-08]],

        [[8.33282948e-01, 1.42837986e-01, 1.27014071e-02,
          1.88645208e-03, 9.25042015e-03, 4.08326341e-05],
         [9.15582597e-01, 8.01559165e-02, 1.87551440e-03,
          1.91104948e-04, 2.19372427e-03, 1.13891008e-06],
         [9.37633574e-01, 5.82904480e-02, 1.90978579e-03,
          8.55049148e-05, 2.08021980e-03, 4.28671115e-07],
         ...,
         [9.86089826e-01, 1.09786531e-02, 1.03561513e-04,
          4.13196494e-06, 2.82380078e-03, 5.43018408e-10],
         [9.71064985e-01, 2.35237945e-02, 2.69633776e-04,
          1.92030911e-05, 5.12242131e-03, 6.85490198e-09],
         [8.92416000e-01, 6.73815906e-02, 4.01798030e-03,
          1.12704898e-03, 3.50522846e-02, 5.02115881e-06]]]],
      dtype=float32)

# ===== Cell Separator =====

array([[[1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 3, 3, 3],
        ...,
        [0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1]],

       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]],

       [[1, 1, 2, ..., 3, 1, 1],
        [1, 2, 2, ..., 3, 1, 1],
        [2, 2, 2, ..., 1, 1, 1],
        ...,
        [1, 1, 1, ..., 2, 2, 2],
        [1, 1, 1, ..., 2, 2, 2],
        [1, 1, 1, ..., 2, 2, 2]],

       ...,

       [[1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        ...,
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]],

       [[0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]],

       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]]])

# ===== Cell Separator =====

array([[[3, 3, 3, ..., 1, 1, 1],
        [3, 3, 3, ..., 3, 3, 3],
        [3, 3, 3, ..., 3, 3, 3],
        ...,
        [0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1],
        [0, 0, 0, ..., 1, 1, 1]],

       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]],

       [[2, 2, 2, ..., 1, 1, 1],
        [2, 2, 2, ..., 1, 1, 1],
        [2, 2, 2, ..., 1, 1, 1],
        ...,
        [2, 2, 2, ..., 0, 0, 0],
        [2, 2, 2, ..., 0, 0, 0],
        [2, 2, 1, ..., 0, 0, 0]],

       ...,

       [[1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        ...,
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]],

       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]],

       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]]])

# ===== Cell Separator =====

test_image_number = random.randint(0, len(X_test))

test_image = X_test[test_image_number]
ground_truth_image = y_test_argmax[test_image_number]

test_image_input = np.expand_dims(test_image, 0)

prediction = model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]

# ===== Cell Separator =====

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(test_image)
plt.subplot(232)
plt.title("Original Masked image")
plt.imshow(ground_truth_image)
plt.subplot(233)
plt.title("Predicted Image")
plt.imshow(predicted_image)

# ===== Cell Separator =====

total 390M
drwxr-x--- 41 user user 4.0K Oct  6 15:09  .
drwxr-xr-x  3 root root 4.0K Apr 16  2024  ..
-rw-rw-r--  1 user user  12K Oct  6 11:48  avalanche_suspectibility.ipynb
-rw-------  1 user user  11K Oct  6 12:46  .bash_history
-rw-r--r--  1 user user  220 Apr 16  2024  .bash_logout
-rw-r--r--  1 user user 3.9K Sep  2 09:17  .bashrc
drwx------ 30 user user 4.0K Oct  1 16:32  .cache
drwxrwxr-x  5 user user 4.0K Sep 18 11:36  catboost_info
drwx------ 28 user user 4.0K Oct  6 14:24  .config
drwxrwxr-x  4 user user 4.0K Sep 29 17:48  deepseek-r1-distill-llama-70b
drwxrwxr-x  3 user user 4.0K Sep 29 10:19  .designer
drwxr-xr-x  2 user user 4.0K Apr 16  2024  Desktop
drwxr-xr-x 10 user user 4.0K Oct  6 09:23  .docker
-rw-rw-r--  1 user user 386M Oct  1 15:44  docker-desktop-amd64.deb
drwxr-xr-x  4 user user 4.0K Oct  6 13:01  Documents
drwxrwxr-x  3 user user 4.0K Sep 15 16:04  .dotnet
drwxr-xr-x 11 user user 4.0K Oct  6 15:39  Downloads
drwxrwxr-x  3 user user 4.0K Sep 11 15:11  .eclipse
-rw-rw-r--  1 user user   40 Sep 30 15:54  .gitconfig
drwx------  2 user user 4.0K Oct  6 10:35  .gnupg
drwxrwxr-x  2 user user 4.0K Sep 17 10:36  .grass7
-rw-rw-r--  1 user user 171K Sep 23 17:01  GUI.ipynb
-rw-rw-r--  1 user user  54K Sep 17 16:37 'image&maks.ipynb'
drwxrwxr-x  2 user user 4.0K Oct  6 11:40  .ipynb_checkpoints
drwxrwxr-x  3 user user 4.0K Sep 10 09:28  .ipython
drwxrwxr-x  3 user user 4.0K Sep 11 10:44  .jupyter
-rw-rw-r--  1 user user 187K Sep 18 12:48  keggle.ipynb
drwxrwxr-x  2 user user 4.0K Sep 23 12:47  .keras
-rw-------  1 user user   20 Sep 30 17:40  .lesshst
drwx------  6 user user 4.0K Sep 10 09:23  .local
-rw-rw-r--  1 user user  38K Sep 24 17:50  metrological_data.ipynb
drwx------  3 user user 4.0K Aug 26 12:30  .mozilla
drwxr-xr-x  2 user user 4.0K Apr 16  2024  Music
drwxrwxr-x  6 user user 4.0K Sep  2 09:36  my-app
-rw-------  1 user user   86 Oct  1 16:23  .netrc
-rw-rw-r--  1 user user  14K Sep 24 16:52  NLP_3.ipynb
drwxrwxr-x  4 user user 4.0K Sep 24 11:12  nltk_data
-rw-rw-r--  1 user user  19K Sep 19 17:45  nltk.ipynb
drwxrwxr-x  5 user user 4.0K Sep  2 09:07  .npm
drwxrwxr-x  8 user user 4.0K Sep  2 09:29  .nvm
drwxrwxr-x  2 user user 4.0K Sep 17 17:14  outputs
drwxrwxr-x  3 user user 4.0K Sep 11 15:11  .p2
drwxr-xr-x  3 user user 4.0K Sep 19 16:58  Pictures
drwx------  3 user user 4.0K Sep  2 09:35  .pki
-rw-r--r--  1 user user  807 Apr 16  2024  .profile
drwxr-xr-x  2 user user 4.0K Apr 16  2024  Public
-rw-------  1 user user   12 Sep  2 11:21  .python_history
-rw-------  1 user user    0 Sep 26 17:53  .python_history-34746.tmp
-rw-rw-r--  1 user user 377K Oct  6 10:29  satellite_model_plot.png
-rw-rw-r--  1 user user 327K Oct  3 17:18  satellite_segmentation_prediction1ipynb
-rw-rw-r--  1 user user 311K Oct  3 16:41  satellite_segmentation_prediction.ipynb
drwxrwxr-x  3 user user 4.0K Sep 10 11:29  scikit_learn_data
-rw-rw-r--  1 user user 171K Sep 11 16:03 'SciKit Learn.ipynb'
drwx------  5 user user 4.0K Aug 26 14:07  snap
drwx------  2 user user 4.0K Aug 28  2024  .ssh
drwxrwxr-x  2 user user 4.0K Sep 26 16:43  .streamlit
-rw-r--r--  1 user user    0 Apr 17  2024  .sudo_as_admin_successful
drwxr-xr-x  2 user user 4.0K Apr 16  2024  Templates
drwx------  6 user user 4.0K Aug 26 12:30  .thunderbird
drwxr-xr-x  3 user user 4.0K Sep 29 14:15  .tuxpaint
-rw-rw-r--  1 user user  41K Sep 17 17:36  unet_model.ipynb
-rw-rw-r--  1 user user 2.8M Oct  6 15:09  unet_segmentation.ipynb
drwxr-xr-x  2 user user 4.0K Apr 16  2024  Videos
drwxrwxr-x  4 user user 4.0K Sep  2 09:35  .vscode
drwxrwxr-x 36 user user 4.0K Oct  6 10:29  wandb

# ===== Cell Separator =====

{'name': 'model',
 'trainable': True,
 'layers': [{'module': 'keras.layers',
   'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 256, 256, 3),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'registered_name': None,
   'name': 'input_1',
   'inbound_nodes': []},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 3)},
   'name': 'conv2d',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'dropout',
   'inbound_nodes': [[['conv2d', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_1',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'conv2d_1',
   'inbound_nodes': [[['dropout', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'max_pooling2d',
   'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_2',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 16)},
   'name': 'conv2d_2',
   'inbound_nodes': [[['max_pooling2d', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_1',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'dropout_1',
   'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_3',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'conv2d_3',
   'inbound_nodes': [[['dropout_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_1',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'max_pooling2d_1',
   'inbound_nodes': [[['conv2d_3', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_4',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 32)},
   'name': 'conv2d_4',
   'inbound_nodes': [[['max_pooling2d_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_2',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'dropout_2',
   'inbound_nodes': [[['conv2d_4', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_5',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'conv2d_5',
   'inbound_nodes': [[['dropout_2', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_2',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'max_pooling2d_2',
   'inbound_nodes': [[['conv2d_5', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_6',
    'trainable': True,
    'dtype': 'float32',
    'filters': 128,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 64)},
   'name': 'conv2d_6',
   'inbound_nodes': [[['max_pooling2d_2', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_3',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'dropout_3',
   'inbound_nodes': [[['conv2d_6', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_7',
    'trainable': True,
    'dtype': 'float32',
    'filters': 128,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'conv2d_7',
   'inbound_nodes': [[['dropout_3', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_3',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'max_pooling2d_3',
   'inbound_nodes': [[['conv2d_7', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_8',
    'trainable': True,
    'dtype': 'float32',
    'filters': 256,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 16, 16, 128)},
   'name': 'conv2d_8',
   'inbound_nodes': [[['max_pooling2d_3', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_4',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 16, 16, 256)},
   'name': 'dropout_4',
   'inbound_nodes': [[['conv2d_8', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_9',
    'trainable': True,
    'dtype': 'float32',
    'filters': 256,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 16, 16, 256)},
   'name': 'conv2d_9',
   'inbound_nodes': [[['dropout_4', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2DTranspose',
   'config': {'name': 'conv2d_transpose',
    'trainable': True,
    'dtype': 'float32',
    'filters': 128,
    'kernel_size': (2, 2),
    'strides': (2, 2),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'linear',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'output_padding': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 16, 16, 256)},
   'name': 'conv2d_transpose',
   'inbound_nodes': [[['conv2d_9', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Concatenate',
   'config': {'name': 'concatenate',
    'trainable': True,
    'dtype': 'float32',
    'axis': -1},
   'registered_name': None,
   'build_config': {'input_shape': [(None, 32, 32, 128), (None, 32, 32, 128)]},
   'name': 'concatenate',
   'inbound_nodes': [[['conv2d_transpose', 0, 0, {}],
     ['conv2d_7', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_10',
    'trainable': True,
    'dtype': 'float32',
    'filters': 128,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 256)},
   'name': 'conv2d_10',
   'inbound_nodes': [[['concatenate', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_5',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'dropout_5',
   'inbound_nodes': [[['conv2d_10', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_11',
    'trainable': True,
    'dtype': 'float32',
    'filters': 128,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'conv2d_11',
   'inbound_nodes': [[['dropout_5', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2DTranspose',
   'config': {'name': 'conv2d_transpose_1',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (2, 2),
    'strides': (2, 2),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'linear',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'output_padding': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 32, 32, 128)},
   'name': 'conv2d_transpose_1',
   'inbound_nodes': [[['conv2d_11', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Concatenate',
   'config': {'name': 'concatenate_1',
    'trainable': True,
    'dtype': 'float32',
    'axis': -1},
   'registered_name': None,
   'build_config': {'input_shape': [(None, 64, 64, 64), (None, 64, 64, 64)]},
   'name': 'concatenate_1',
   'inbound_nodes': [[['conv2d_transpose_1', 0, 0, {}],
     ['conv2d_5', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_12',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 128)},
   'name': 'conv2d_12',
   'inbound_nodes': [[['concatenate_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_6',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'dropout_6',
   'inbound_nodes': [[['conv2d_12', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_13',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'conv2d_13',
   'inbound_nodes': [[['dropout_6', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2DTranspose',
   'config': {'name': 'conv2d_transpose_2',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (2, 2),
    'strides': (2, 2),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'linear',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'output_padding': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 64, 64, 64)},
   'name': 'conv2d_transpose_2',
   'inbound_nodes': [[['conv2d_13', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Concatenate',
   'config': {'name': 'concatenate_2',
    'trainable': True,
    'dtype': 'float32',
    'axis': -1},
   'registered_name': None,
   'build_config': {'input_shape': [(None, 128, 128, 32),
     (None, 128, 128, 32)]},
   'name': 'concatenate_2',
   'inbound_nodes': [[['conv2d_transpose_2', 0, 0, {}],
     ['conv2d_3', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_14',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 64)},
   'name': 'conv2d_14',
   'inbound_nodes': [[['concatenate_2', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_7',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'dropout_7',
   'inbound_nodes': [[['conv2d_14', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_15',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'conv2d_15',
   'inbound_nodes': [[['dropout_7', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2DTranspose',
   'config': {'name': 'conv2d_transpose_3',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (2, 2),
    'strides': (2, 2),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'linear',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'output_padding': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 128, 128, 32)},
   'name': 'conv2d_transpose_3',
   'inbound_nodes': [[['conv2d_15', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Concatenate',
   'config': {'name': 'concatenate_3',
    'trainable': True,
    'dtype': 'float32',
    'axis': 3},
   'registered_name': None,
   'build_config': {'input_shape': [(None, 256, 256, 16),
     (None, 256, 256, 16)]},
   'name': 'concatenate_3',
   'inbound_nodes': [[['conv2d_transpose_3', 0, 0, {}],
     ['conv2d_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_16',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 32)},
   'name': 'conv2d_16',
   'inbound_nodes': [[['concatenate_3', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout_8',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'dropout_8',
   'inbound_nodes': [[['conv2d_16', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_17',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'HeNormal',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'conv2d_17',
   'inbound_nodes': [[['dropout_8', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Conv2D',
   'config': {'name': 'conv2d_18',
    'trainable': True,
    'dtype': 'float32',
    'filters': 6,
    'kernel_size': (1, 1),
    'strides': (1, 1),
    'padding': 'valid',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'module': 'keras.initializers',
     'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'registered_name': None},
    'bias_initializer': {'module': 'keras.initializers',
     'class_name': 'Zeros',
     'config': {},
     'registered_name': None},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, 256, 256, 16)},
   'name': 'conv2d_18',
   'inbound_nodes': [[['conv2d_17', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['conv2d_18', 0, 0]]}

# ===== Cell Separator =====

test_image_number = random.randint(0, len(X_test))

test_image = X_test[test_image_number]
ground_truth_image = y_test_argmax[test_image_number]

test_image_input = np.expand_dims(test_image, 0)

prediction = save_model.predict(test_image_input)
predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]

# ===== Cell Separator =====

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(test_image)
plt.subplot(232)
plt.title("Original Masked image")
plt.imshow(ground_truth_image)
plt.subplot(233)
plt.title("Predicted Image")
plt.imshow(predicted_image)

# ===== Cell Separator =====

image = Image.open('/home/user/Downloads/img3.jpg')
image = image.resize((256,256))
image = np.array(image)
image = np.expand_dims(image, 0)

# ===== Cell Separator =====

array([[[[135, 120, 123],
         [110,  99,  99],
         [119, 111, 108],
         ...,
         [ 41,  57,  73],
         [ 46,  60,  92],
         [ 29,  35,  71]],

        [[136, 123, 125],
         [118, 108, 107],
         [118, 113, 110],
         ...,
         [ 57,  74,  93],
         [ 38,  49,  83],
         [ 26,  32,  67]],

        [[158, 146, 142],
         [132, 122, 116],
         [104,  98,  95],
         ...,
         [ 64,  81, 103],
         [ 29,  35,  70],
         [ 27,  33,  68]],

        ...,

        [[ 35,  52,  60],
         [ 20,  37,  45],
         [ 17,  31,  40],
         ...,
         [ 40,  51,  62],
         [ 50,  60,  70],
         [ 58,  69,  75]],

        [[ 19,  33,  43],
         [ 14,  28,  38],
         [ 21,  33,  43],
         ...,
         [ 45,  56,  70],
         [ 55,  68,  79],
         [ 57,  70,  78]],

        [[ 31,  42,  54],
         [ 16,  27,  39],
         [ 29,  38,  51],
         ...,
         [ 24,  35,  53],
         [ 71,  82,  97],
         [ 61,  74,  84]]]], dtype=uint8)

# ===== Cell Separator =====

predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]

# ===== Cell Separator =====

plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(test_image)
plt.subplot(232)
plt.title("Original Masked image")
plt.imshow(ground_truth_image)
plt.subplot(233)
plt.title("Predicted Image")
plt.imshow(predicted_image)

# ===== Cell Separator =====

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: keract in ./.local/lib/python3.10/site-packages (4.5.2)

# ===== Cell Separator =====

total 58M
drwxrwxr-x 2 user user 4.0K Oct  3 15:29 ./
drwxrwxr-x 6 user user 4.0K Oct  6 15:25 ../
-rw-rw-r-- 1 user user 2.5M Oct  3 16:13 0_input_1.png
-rw-rw-r-- 1 user user 1.7M Oct  3 16:13 10_dropout_2.png
-rw-rw-r-- 1 user user 701K Oct  3 16:13 11_conv2d_5.png
-rw-rw-r-- 1 user user 107K Oct  3 16:13 12_max_pooling2d_2.png
-rw-rw-r-- 1 user user 153K Oct  3 16:13 13_conv2d_6.png
-rw-rw-r-- 1 user user 153K Oct  3 16:13 14_dropout_3.png
-rw-rw-r-- 1 user user  84K Oct  3 16:13 15_conv2d_7.png
-rw-rw-r-- 1 user user  59K Oct  3 16:13 16_max_pooling2d_3.png
-rw-rw-r-- 1 user user  87K Oct  3 16:13 17_conv2d_8.png
-rw-rw-r-- 1 user user  87K Oct  3 16:13 18_dropout_4.png
-rw-rw-r-- 1 user user  90K Oct  3 16:13 19_conv2d_9.png
-rw-rw-r-- 1 user user 2.8M Oct  3 16:13 1_conv2d.png
-rw-rw-r-- 1 user user 585K Oct  3 16:14 20_conv2d_transpose.png
-rw-rw-r-- 1 user user 2.1M Oct  3 16:14 21_concatenate.png
-rw-rw-r-- 1 user user 113K Oct  3 16:14 22_conv2d_10.png
-rw-rw-r-- 1 user user 113K Oct  3 16:14 23_dropout_5.png
-rw-rw-r-- 1 user user 148K Oct  3 16:14 24_conv2d_11.png
-rw-rw-r-- 1 user user 3.5M Oct  3 16:14 25_conv2d_transpose_1.png
-rw-rw-r-- 1 user user 2.5M Oct  3 16:14 26_concatenate_1.png
-rw-rw-r-- 1 user user 400K Oct  3 16:14 27_conv2d_12.png
-rw-rw-r-- 1 user user 400K Oct  3 16:14 28_dropout_6.png
-rw-rw-r-- 1 user user 1.3M Oct  3 16:14 29_conv2d_13.png
-rw-rw-r-- 1 user user 2.8M Oct  3 16:13 2_dropout.png
-rw-rw-r-- 1 user user 2.8M Oct  3 16:14 30_conv2d_transpose_2.png
-rw-rw-r-- 1 user user 2.8M Oct  3 16:14 31_concatenate_2.png
-rw-rw-r-- 1 user user 2.0M Oct  3 16:14 32_conv2d_14.png
-rw-rw-r-- 1 user user 2.0M Oct  3 16:14 33_dropout_7.png
-rw-rw-r-- 1 user user 860K Oct  3 16:14 34_conv2d_15.png
-rw-rw-r-- 1 user user 4.7M Oct  3 16:14 35_conv2d_transpose_3.png
-rw-rw-r-- 1 user user 3.8M Oct  3 16:14 36_concatenate_3.png
-rw-rw-r-- 1 user user 1.4M Oct  3 16:14 37_conv2d_16.png
-rw-rw-r-- 1 user user 1.4M Oct  3 16:14 38_dropout_8.png
-rw-rw-r-- 1 user user 1.4M Oct  3 16:14 39_conv2d_17.png
-rw-rw-r-- 1 user user 3.5M Oct  3 16:13 3_conv2d_1.png
-rw-rw-r-- 1 user user  56K Oct  3 16:14 40_conv2d_18.png
-rw-rw-r-- 1 user user 3.0M Oct  3 16:13 4_max_pooling2d.png
-rw-rw-r-- 1 user user 1.5M Oct  3 16:13 5_conv2d_2.png
-rw-rw-r-- 1 user user 1.5M Oct  3 16:13 6_dropout_1.png
-rw-rw-r-- 1 user user 1.7M Oct  3 16:13 7_conv2d_3.png
-rw-rw-r-- 1 user user 342K Oct  3 16:13 8_max_pooling2d_1.png
-rw-rw-r-- 1 user user 1.7M Oct  3 16:13 9_conv2d_4.png

# ===== Cell Separator =====

input_1 (1, 256, 256, 3)
conv2d (1, 256, 256, 16)
dropout (1, 256, 256, 16)
conv2d_1 (1, 256, 256, 16)
max_pooling2d (1, 128, 128, 16)
conv2d_2 (1, 128, 128, 32)
dropout_1 (1, 128, 128, 32)
conv2d_3 (1, 128, 128, 32)
max_pooling2d_1 (1, 64, 64, 32)
conv2d_4 (1, 64, 64, 64)
dropout_2 (1, 64, 64, 64)
conv2d_5 (1, 64, 64, 64)
max_pooling2d_2 (1, 32, 32, 64)
conv2d_6 (1, 32, 32, 128)
dropout_3 (1, 32, 32, 128)
conv2d_7 (1, 32, 32, 128)
max_pooling2d_3 (1, 16, 16, 128)
conv2d_8 (1, 16, 16, 256)
dropout_4 (1, 16, 16, 256)
conv2d_9 (1, 16, 16, 256)
conv2d_transpose (1, 32, 32, 128)
concatenate (1, 32, 32, 256)
conv2d_10 (1, 32, 32, 128)
dropout_5 (1, 32, 32, 128)
conv2d_11 (1, 32, 32, 128)
conv2d_transpose_1 (1, 64, 64, 64)
concatenate_1 (1, 64, 64, 128)
conv2d_12 (1, 64, 64, 64)
dropout_6 (1, 64, 64, 64)
conv2d_13 (1, 64, 64, 64)
conv2d_transpose_2 (1, 128, 128, 32)
concatenate_2 (1, 128, 128, 64)
conv2d_14 (1, 128, 128, 32)
dropout_7 (1, 128, 128, 32)
conv2d_15 (1, 128, 128, 32)
conv2d_transpose_3 (1, 256, 256, 16)
concatenate_3 (1, 256, 256, 32)
conv2d_16 (1, 256, 256, 16)
dropout_8 (1, 256, 256, 16)
conv2d_17 (1, 256, 256, 16)
conv2d_18 (1, 256, 256, 6)

# ===== Cell Separator =====

image = Image.open('/home/user/Downloads/img3.jpg')
image = image.resize((256,256))
image_as_array = np.array(image)
image_as_array = image_as_array.astype(np.float32)
ke.display_heatmaps(activations, image_as_array, save=True, directory= '/home/user/Downloads/archive/heatmap')

# ===== Cell Separator =====

input_1 (1, 256, 256, 3)
conv2d (1, 256, 256, 16)
dropout (1, 256, 256, 16)
conv2d_1 (1, 256, 256, 16)
max_pooling2d (1, 128, 128, 16)
conv2d_2 (1, 128, 128, 32)
dropout_1 (1, 128, 128, 32)
conv2d_3 (1, 128, 128, 32)
max_pooling2d_1 (1, 64, 64, 32)
conv2d_4 (1, 64, 64, 64)
dropout_2 (1, 64, 64, 64)
conv2d_5 (1, 64, 64, 64)
max_pooling2d_2 (1, 32, 32, 64)
conv2d_6 (1, 32, 32, 128)
dropout_3 (1, 32, 32, 128)
conv2d_7 (1, 32, 32, 128)
max_pooling2d_3 (1, 16, 16, 128)
conv2d_8 (1, 16, 16, 256)
dropout_4 (1, 16, 16, 256)
conv2d_9 (1, 16, 16, 256)
conv2d_transpose (1, 32, 32, 128)
concatenate (1, 32, 32, 256)
conv2d_10 (1, 32, 32, 128)
dropout_5 (1, 32, 32, 128)
conv2d_11 (1, 32, 32, 128)
conv2d_transpose_1 (1, 64, 64, 64)
concatenate_1 (1, 64, 64, 128)
conv2d_12 (1, 64, 64, 64)
dropout_6 (1, 64, 64, 64)
conv2d_13 (1, 64, 64, 64)
conv2d_transpose_2 (1, 128, 128, 32)
concatenate_2 (1, 128, 128, 64)
conv2d_14 (1, 128, 128, 32)
dropout_7 (1, 128, 128, 32)
conv2d_15 (1, 128, 128, 32)
conv2d_transpose_3 (1, 256, 256, 16)
concatenate_3 (1, 256, 256, 32)
conv2d_16 (1, 256, 256, 16)
dropout_8 (1, 256, 256, 16)
conv2d_17 (1, 256, 256, 16)
conv2d_18 (1, 256, 256, 6)

# ===== Cell Separator =====

import wandb

wandb.init(project="satellite-image-dubai-data", name="dry-leaf-18")

# after training
wandb.log_artifact("model.h5", type="model", name="dry-leaf-18-model")

# ===== Cell Separator =====

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[171], line 6
wandb.init(project="satellite-image-dubai-data", name="dry-leaf-18")
# after training
----> 6 wandb.log_artifact("model.h5", type="model", name="dry-leaf-18-model")

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:399, in _log_to_run.<locals>.wrapper(self, *args, **kwargs)
    run_id = self._attach_id
with wb_logging.log_to_run(run_id):
--> 399     return func(self, *args, **kwargs)

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:457, in _raise_if_finished.<locals>.wrapper_fn(self, *args, **kwargs)
@functools.wraps(func)
def wrapper_fn(self: Run, *args, **kwargs) -> _T:
    if not getattr(self, "_is_finished", False):
--> 457         return func(self, *args, **kwargs)
    message = (
        f"Run ({self.id}) is finished. The call to"
        f" `{func.__name__}` will be ignored."
        f" Please make sure that you are using an active run."
    )
    raise UsageError(message)

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:444, in _attach.<locals>.wrapper(self, *args, **kwargs)
    finally:
        _is_attaching = ""
--> 444 return func(self, *args, **kwargs)

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:3172, in Run.log_artifact(self, artifact_or_path, name, type, aliases, tags)
@_log_to_run
@_raise_if_finished
@_attach
   (...)
    tags: list[str] | None = None,
) -> Artifact:
    """Declare an artifact as an output of a run.

    Args:
   (...)
        An `Artifact` object.
    """
-> 3172     return self._log_artifact(
        artifact_or_path,
        name=name,
        type=type,
        aliases=aliases,
        tags=tags,
    )

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:3313, in Run._log_artifact(self, artifact_or_path, name, type, aliases, tags, distributed_id, finalize, is_user_created, use_after_commit)
if tags is not None:
    tags = validate_tags(tags)
-> 3313 artifact, aliases = self._prepare_artifact(
    artifact_or_path, name, type, aliases
)
if len(artifact.metadata) > MAX_ARTIFACT_METADATA_KEYS:
    raise ValueError(
        f"Artifact must not have more than {MAX_ARTIFACT_METADATA_KEYS} metadata keys."
    )

File ~/.local/lib/python3.10/site-packages/wandb/sdk/wandb_run.py:3418, in Run._prepare_artifact(self, artifact_or_path, name, type, aliases)
        artifact.add_reference(str(artifact_or_path))
    else:
-> 3418         raise ValueError(
            "path must be a file, directory or external"
            "reference like s3://bucket/path"
        )
else:
    artifact = artifact_or_path

ValueError: path must be a file, directory or externalreference like s3://bucket/path
