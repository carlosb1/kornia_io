import torch
import kornia_rust as K_rs
import cv2

data, shape = K_rs.read_image("/home/edgar/Downloads/IMG_20211219_145924.jpg")
img_t = torch.tensor(data, dtype=torch.uint8).reshape(shape)
print(img_t.shape)

cv2.imshow("rust_image", img_t[..., (2,1,0)].numpy())
cv2.waitKey(0)
