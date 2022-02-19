from distutils import extension
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
from torch import Tensor

import kornia as K
import kornia_rs


class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    @staticmethod 
    def __new__(cls, data, color, *args, **kwargs): 
        return super().__new__(cls, data, *args, **kwargs) 

    def __init__(self, data: Tensor, color: ImageColor) -> None:
        self._color = color

    @property
    def valid(self) -> bool:
        # TODO: find a better way to do this
        return self.data.data_ptr is not None

    @property
    def is_batch(self) -> bool:
        return len(self.data.shape) > 3

    @property
    def channels(self) -> int:
        return self.data.shape[-3]

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.data.shape[-2:]

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._color

    @color.setter 
    def color(self, c: ImageColor):
        self._color = c 

    @classmethod
    def from_tensor(cls, data: Tensor, color: ImageColor = ImageColor.RGB) -> 'Image':
        return cls(data, color)

    @classmethod
    def from_numpy(cls, data: np.ndarray, color: ImageColor = ImageColor.RGB) -> 'Image':
        data_t: Tensor = K.utils.image_to_tensor(data).float()
        return cls(data_t, color)

    def to_numpy(self) -> np.ndarray:
        return K.utils.tensor_to_image(self.data, keepdim=True)

    @classmethod
    def from_list(cls, data: List[List[Union[float, int]]], color: ImageColor = ImageColor.RGB) -> 'Image':
        data_t: Tensor = Tensor(data)
        return cls(data_t, color)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        #data: np.ndarray = __image_reader__(file_path)
        #data_t: Tensor = K.utils.image_to_tensor(data)
        data_t: Tensor = __image_reader__(file_path)
        data_t = K.color.bgr_to_rgb(data_t)
        # TODO: discuss whether we return normalised
        data_t = data_t.float() / 255.
        return cls(data_t, ImageColor.RGB)

    def _to_grayscale(self, data: Tensor) -> Tensor:
        if self.color == ImageColor.GRAY:
            out = data
        elif self.color == ImageColor.RGB:
            out = K.color.rgb_to_grayscale(data)
        elif self.color == ImageColor.BGR:
            out = K.color.bgr_to_grayscale(data)
        else:
            raise NotImplementedError(f"Unsupported color: {self.color}.")
        return out

    def grayscale(self) -> 'Image':
        gray = self._to_grayscale(self.data)
        return Image(gray, ImageColor.GRAY)

    def grayscale_(self) -> 'Image':
        self.data = self._to_grayscale(self.data)
        self._color = ImageColor.GRAY
        return self

    def _convert(self, color: ImageColor) -> Tuple[Tensor, ImageColor]:
        if color == ImageColor.RGB:
            if self.color == ImageColor.BGR:
                data_out = K.color.bgr_to_rgb(self.data)
                color_out = ImageColor.RGB
        elif color == ImageColor.BGR:
            if self.color == ImageColor.RGB:
                data_out = K.color.rgb_to_bgr(self.data)
                color_out = ImageColor.BGR
        elif color == ImageColor.GRAY:
            if self.color == ImageColor.BGR:
                data_out = K.color.bgr_to_grayscale(self.data)
                color_out = ImageColor.GRAY
            elif self.color == ImageColor.RGB:
                data_out = K.color.rgb_to_grayscale(self.data)
                color_out = ImageColor.GRAY
        else:
            raise NotImplementedError(f"Unsupported color: {self.color}.")
        return data_out, color_out

    def convert(self, color: ImageColor) -> 'Image':
        data, color = self._convert(color)
        return Image(data, color)

    def convert_(self, color: ImageColor) -> 'Image':
        self.data, self.color = self._convert(color)
        return self

    def hflip(self) -> 'Image':
        data = K.geometry.transform.hflip(self.data)
        return Image(data, self.color)

    def vflip(self) -> 'Image':
        data = K.geometry.transform.vflip(self.data)
        return Image(data, self.color)

    # TODO: add the methods we need
    # - lab, hsv, ...
    # - erode, dilate, ...


## READING API

# TODO: implement Image class here
def read_image(file_path: str, device: Optional[torch.device] = None) -> Tensor:
    # TODO: implement extension with pathlib
    extension: str = file_path.split('.')[-1]
    if extension == "jpg":
        # use libjpeg-turbo for best performance
        data, shape = kornia_rs.read_image_jpeg(file_path)
    else:
        # use image-rs for general image decoding
        data, shape = kornia_rs.read_image(file_path)
    # cast to tensor and device, data comes in HxWxC
    img_t = torch.as_tensor(data, device=device, dtype=torch.uint8)
    return img_t.reshape(shape).permute(2, 1, 0)  # CxHxW


def read_image_dlpack(file_path: str, device: Optional[torch.device] = None) -> Tensor:
    # import pdb;pdb.set_trace()
    dl_tensor = kornia_rs.read_image_dlpack(file_path)
    return torch.utils.dlpack.from_dlpack(dl_tensor)


def show_image(_input: Union[str, Tensor]) -> None:
    if isinstance(_input, str):
        file_path: str = _input
        return kornia_rs.show_image_from_file(file_path)
    elif isinstance(_input, Tensor):
        img = _input
        if img.shape[-3] == 1:
            img = img.repeat(3, 1, 1)
        if img.dtype == torch.float:
            img = img.mul_(255.).byte()

        img = img.permute(2, 1, 0)  # CxHxW -> HxWxC
        data = img.cpu().reshape(-1).tolist()  # flatten (HxWxC)
        return kornia_rs.show_image_from_raw(data, img.shape)
    return