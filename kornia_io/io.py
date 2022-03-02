from typing import Optional, Union
from pathlib import Path
from enum import Enum
from kornia_io.core.image import ImageLayout

import kornia_rs
from kornia_rs import Tensor as cvTensor
from kornia_io.core import Image, ImageColor

import torch

class Device(Enum):
    CPU = 0
    CUDA = 1

    def from_string(mode: Optional[str] = None) -> Optional[torch.device]:
        if mode == "cpu":
            return torch.device("cpu")
        elif mode == "cuda":
            return torch.device("cuda")
        elif mode is None:
            return None
        else:
            raise NotImplementedError(f"Unsupported device: {mode}.")


def read_image(file_path: str, device: Optional[str] = None) -> Image:
    cv_tensor: cvTensor
    extension: str = Path(file_path).suffix
    if extension == ".jpg":
        # use libjpeg-turbo for best performance
        cv_tensor = kornia_rs.read_image_jpeg(file_path)
    else:
        # use image-rs for general image decoding
        cv_tensor = kornia_rs.read_image_rs(file_path)
    # cast to tensor and device, data comes in WxHxC
    dl_tensor = kornia_rs.cvtensor_to_dlpack(cv_tensor)
    th_tensor = torch.utils.dlpack.from_dlpack(dl_tensor)
    return Image(
        th_tensor.squeeze(), ImageColor.RGB, ImageLayout.HWC).to_chw()
