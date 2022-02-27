import kornia as K
import kornia_io as Kx
from kornia_io.core import Image

file_path = "/home/edgar/Downloads/IMG_20211219_145924.jpg"

# load the image with libjpeg-turbo and send directly to the gpu
img: Image = Kx.io.read_image(file_path, device="cpu")
# NOTE: uncomment below to use a different api
# img =  Image.from_file(file_path, device="cuda")
img_src: Image = img.clone()  # for vis
print(f"Image: {img.shape}, dtype: {img.dtype}, device: {img.device}")

# do something with and show
img = img[None].float() / 255.  # normalize
img = K.color.rgb_to_grayscale(img)
img = K.contrib.distance_transform(img)
img = img.apply(K.geometry.resize, (224, 224))
# NOTE: uncomment below to perform in-place
# img.apply_(K.geometry.resize, (224, 224))
assert isinstance(img, Image)  # type it's propagated :)

# show using our cool rust-opengl (vviz lib :)
opts = Kx.viz.VizOptions(denormalize=True)
Kx.viz.show_image("original", img_src)
Kx.viz.show_image("distance", img, opts)

# TODO(edgar/hauke): coming soon
# viz = Kx.viz.VizManager()
# viz.add_image("original", image1)
# viz.add_image("distance", image2)
# viz.show()
