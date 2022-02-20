from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass

from kornia_io.core.image import Image

import kornia_rs
from kornia_rs import Tensor as cvTensor


@dataclass
class VizOptions:
    denormalize: bool = False


def show_image(window_name: str, data: Union[Image, Path, str], opts: Optional[VizOptions] = None):
    if opts is None:
        opts = VizOptions()

    if isinstance(data, (Path, str,)):
        file_path: Path = Path(data)
        kornia_rs.show_image_from_file(file_path)
    elif isinstance(data, Image):
        image: Image = data
        cv_img: cvTensor = image.to_viz(denormalize=opts.denormalize)
        kornia_rs.show_image_from_tensor(window_name, cv_img)

# TODO: implement VizManager wrapper
# VizManager = kornia_rs.VizManager
