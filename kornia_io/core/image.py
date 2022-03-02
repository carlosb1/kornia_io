from enum import Enum
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path

from kornia.utils import image_to_tensor, tensor_to_image

from kornia_io.core import Tensor
from kornia_rs import Tensor as cvTensor

class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2

class ImageLayout(Enum):
    CHW = 0
    HWC = 1

class Image(Tensor):
    _color = ImageColor.RGB
    _layout = ImageLayout.HWC

    @staticmethod
    def __new__(cls, data, color, layout, *args, **kwargs):
        return Tensor._make_subclass(cls, data)

    def __init__(self, data: Tensor, color: ImageColor, layout: ImageLayout) -> None:
        self._color = color
        self._layout = layout

    @property
    def valid(self) -> bool:
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
        return tuple(self.data.shape[-2:])

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._color

    @property
    def layout(self) -> ImageLayout:
        return self._layout

    @classmethod
    def from_numpy(cls, data, color: ImageColor = ImageColor.RGB) -> 'Image':
        return cls(image_to_tensor(data), color)

    def to_numpy(self):
        return tensor_to_image(self.data, keepdim=True)

    def to_tensor(self, denormalize: bool) -> cvTensor:
        img: Tensor = self.data
        if denormalize:
            img = img.mul_(255)
        img = img.squeeze_(0).byte()
        # handle the grayscale case and replicate the channel
        # to fulfill rgb requirements.
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        # CxHxW => HxWxC
        img = img.permute(1, 2, 0).cpu()
        # TODO: implement cvTensor.from_dlpack(_img.to_dlpack())
        return cvTensor(img.shape, img.reshape(-1).tolist())

    def to_chw(self) -> 'Image':
        if self.layout is ImageLayout.CHW:
            return self
        return self.permute(2, 0, 1)

    def to_hwc(self) -> 'Image':
        if self.layout is ImageLayout.HWC:
            return self
        return self.permute(1, 2, 0)

    # TODO: possibly call torch.as_tensor
    @classmethod
    def from_list(cls, data: List[List[Union[float, int]]], color: ImageColor) -> 'Image':
        return cls(Tensor(data), color)

    @classmethod
    def from_file(cls, file_path: Path, device: Optional[str] = None) -> 'Image':
        from kornia_io import io
        return io.read_image(file_path, device)

    def apply_(self, handle: Callable, *args, **kwargs) -> None:
        self.data = handle(self.data, *args, **kwargs)

    def apply(self, handle: Callable, *args, **kwargs) -> 'Image':
        return Image(handle(self.data, *args, **kwargs), self.color, self.layout)

    # TODO: add the methods we need
    # - grayscale, ..
    # - lab, hsv, ...
    # - erode, dilate, ...
