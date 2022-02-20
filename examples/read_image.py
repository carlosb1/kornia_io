import torch
from torch import Tensor

import kornia as K
import kornia_io as K_io
import kornia_rs as K_rs
from kornia_rs import Tensor as cvTensor

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

# load the image directly in the gpu
img = K_io.read_image(file_path, device=torch.device("cpu"))
img_src = img.clone()  # for vis
print(f"Image: {img.shape}, dtype: {img.dtype}, device: {img.device}")

# do something with cuda and show
img = K.color.rgb_to_grayscale(img[None].float() / 255.)
img = K.contrib.distance_transform(img)

# show using our cool rust-opengl (vviz lib :)

# TODO: make this function more consistant, ans possibly inside Image
def to_viz(_img: Tensor, denorm: bool):
    if denorm:
        _img = _img.mul_(255)
    _img = _img.squeeze_(0).byte()
    if _img.shape[0] == 1:
        _img = _img.repeat(3, 1, 1)
    # CxHxW => WxHxC
    _img = _img.permute(2, 1, 0).contiguous().cpu()
    return _img.reshape(-1).tolist(), _img.shape

img1_vis, shape1 = to_viz(img_src, denorm=False)
img2_vis, shape2 = to_viz(img, denorm=True)

#TODO: error -> OverflowError: out of range integral type conversion attempted
# tensor = cvTensor(img1_vis, shape1)

K_rs.show_image_from_raw(img1_vis, shape1)

#viz = K_rs.VizManager()
#viz.add_image("original", img1_vis, shape1)
#viz.add_image("distance", img2_vis, shape2)
#viz.show()
