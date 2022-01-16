import torch
import kornia_io as K
import cv2

img_t = K.read_image("/home/edgar/Downloads/IMG_20211219_145924.jpg")
print(img_t.shape)

cv2.imshow("rust_image", img_t[..., (2,1,0)].numpy())
cv2.waitKey(0)
