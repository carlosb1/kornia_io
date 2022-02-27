from typing import Optional, Union
from dataclasses import dataclass

import visdom
from kornia_io.core import Tensor
from kornia_io.core.image import Image

import kornia_rs
from kornia_rs import Tensor as cvTensor

# TODO(edgar/hauke): implement VizManager wrapper
# VizManager = kornia_rs.VizManager

@dataclass
class VizOptions:
    denormalize: bool = False


def show_image(window_name: str, data: Union[Image, str], opts: Optional[VizOptions] = None):
    if opts is None:
        opts = VizOptions()

    if isinstance(data, str):
        file_path: str = data
        kornia_rs.show_image_from_file(window_name, file_path)
    elif isinstance(data, Image):
        image: Image = data
        cv_img: cvTensor = image.to_tensor(denormalize=opts.denormalize)
        kornia_rs.show_image_from_tensor(window_name, cv_img)


class VisdomManager:
    def __init__(self, port: int = 8097) -> None:
        super().__init__()
        self._port = port

        self._manager = visdom.Visdom(port=self._port, raise_exceptions=False)

        if not self._manager.check_connection():
            raise ConnectionError(
                f"Error connecting with the visdom server. Run in your termnal: visdom -port {self._port}.")

    def add_image(self, window_name: str, image: Image) -> None:

        if not isinstance(image, (Image, Tensor,)):
            image = Image.from_numpy(image)

        if len(image.shape) == 4:
            self._manager.images(image, win=window_name)
        elif len(image.shape) in (2, 3,):
            self._manager.image(image, win=window_name)
        else:
            raise NotImplementedError(f"Unsupported image size. Got: {image.shape}.")