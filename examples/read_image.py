import torch
from torch import Tensor

import kornia as K
import kornia_io as K_io
import kornia_rs as K_rs
from kornia_rs import Tensor as cvTensor

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

# load the image with libjpeg-turbo and send directly to the gpu
img = K_io.read_image(file_path, device=torch.device("cpu"))
img_src = img.clone()  # for vis
print(f"Image: {img.shape}, dtype: {img.dtype}, device: {img.device}")

# do something with cuda and show
img = K.color.rgb_to_grayscale(img[None].float() / 255.)
img = K.contrib.distance_transform(img)

# show using our cool rust-opengl (vviz lib :)

# TODO: make this function more consistant, ans possibly inside Image
# TODO: implement Image.to_viz()
def to_viz(_img: Tensor, denorm: bool) -> cvTensor:
    if denorm:
        _img = _img.mul_(255)
    _img = _img.squeeze_(0).byte()
    if _img.shape[0] == 1:
        _img = _img.repeat(3, 1, 1)
    # CxHxW => WxHxC
    _img = _img.permute(2, 1, 0).cpu()
    # TODO: implement cvTensor.from_dlpack(_img.to_dlpack())
    return cvTensor(_img.shape, _img.reshape(-1).tolist())

image1 = to_viz(img_src, denorm=False)
image2 = to_viz(img_src, denorm=True)
print(image1.shape)
print(image2.shape)

K_rs.show_image_from_raw(image1)
#K_rs.show_image_from_raw(image1.data, image1.shape)

#viz = K_rs.VizManager()
#viz.add_image("original", image1)
#viz.add_image("distance", image2)
#viz.show()
