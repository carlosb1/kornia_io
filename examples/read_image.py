import torch
import kornia as K
import kornia_io as K_io

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

img_t = K_io.read_image(file_path, device=torch.device("cpu"))
print(img_t.shape)
print(img_t.device)

# do something with cuda and show
img_t = K.color.rgb_to_grayscale(img_t.float() / 255.)

# show using our cool rust-opencl (vviz lib :)
K_io.show_image(img_t.repeat(3, 1, 1).mul(255.).byte())
