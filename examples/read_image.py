import torch

import kornia as K
import kornia_io as K_io

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

# load the image directly in the gpu
img = K_io.read_image(file_path, device=torch.device("cpu"))
img_src = img.clone()  # for vis
print(f"Image: {img.shape}, dtype: {img.dtype}, device: {img.device}")

# do something with cuda and show
img = K.color.rgb_to_grayscale(img[None].float() / 255.)
img = K.contrib.distance_transform(img)

# show using our cool rust-opencl (vviz lib :)
K_io.show_image(img_src)
K_io.show_image(img.squeeze(0))
