import time
import torch
from pathlib import Path
import kornia_rs
import torchvision
import cv2


def test1_io():
    resolved_path: Path = Path(__file__).parent.resolve() / "test.jpg"
    print(resolved_path)
    start = time.perf_counter()
    data, shape = kornia_rs.read_image(str(resolved_path))
    end = time.perf_counter()
    start_2 = time.perf_counter()
    img_t = torch.tensor(data, dtype=torch.uint8).reshape(shape)
    end_2 = time.perf_counter()
    print(
        f'rust: path: {resolved_path} read time: {end-start} secs, '
        f'set up tensor time {end_2 - start_2} secs, '
        f'shape: {img_t.shape}')


def test1_pytorchvision():
    resolved_path: Path = Path(__file__).parent.resolve() / "test.jpg"
    start = time.perf_counter()
    img_t = torchvision.io.read_image(str(resolved_path))
    end = time.perf_counter()
    print(
        f'pytorch code: path: {resolved_path} diff: {end-start} secs, '
        f'shape: {img_t.shape}')


if __name__ == '__main__':
    test1_io()
    test1_pytorchvision()
