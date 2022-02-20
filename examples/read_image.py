import torch

import kornia as K
import kornia_io as K_io
from kornia_io import Image

import kornia_rs as K_rs
from kornia_rs import Tensor as cvTensor

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

# load the image with libjpeg-turbo and send directly to the gpu
img: Image = K_io.read_image(file_path, device="cuda")
# uncomment below to do the same
# img =  Image.from_file(file_path, device="cuda")
img_src: Image = img.clone()  # for vis
print(f"Image: {img.shape}, dtype: {img.dtype}, device: {img.device}")

# do something with and show
img = K.color.rgb_to_grayscale(img[None].float() / 255.)
img = K.contrib.distance_transform(img)
assert isinstance(img, Image)  # type it's propagated :)

# show using our cool rust-opengl (vviz lib :)
image1: cvTensor = img.to_viz(denorm=True)
image2: cvTensor = img_src.to_viz(denorm=False)
print(image1.shape)
print(image2.shape)

K_rs.show_image_from_raw(image1)
K_rs.show_image_from_raw(image2)

# TODO: this 
#viz = K_rs.VizManager()
#viz.add_image("original", image1)
#viz.add_image("distance", image2)
#viz.show()
