from typing import Optional, Union
from pathlib import Path
from enum import Enum

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
    # TODO: implement from dlpack
    import pdb;pdb.set_trace()
    dl_tensor = kornia_rs.cvtensor_to_dlpack(cv_tensor)
    th_tensor = torch.utils.dlpack.from_dlpack(dl_tensor)
    return Image.from_tensor(th_tensor, ImageColor.RGB)
    pass
    #device_ = Device.from_string(device)
    #data = torch.as_tensor(tensor.data, device=device_, dtype=torch.uint8)
    #img_t = Image.from_tensor(data, ImageColor.RGB)
    #return img_t.reshape(tensor.shape).permute(2, 0, 1)  # HxWxC->CxHxW
