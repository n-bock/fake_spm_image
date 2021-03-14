"""Create fake SPM images by providing input image.

    (C) Copyright Nicolas Bock, licensed under GPL v3
    See LICENSE or http://www.gnu.org/licenses/gpl-3.0.html
"""

import fire
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, transform


def add_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape)
    return img + noise


def blur_image(img, px_size):
    return transform.resize(img, (px_size, px_size))


def add_line_distortion(img, intensity):
    for i in range(img.shape[0]):
        img[i, :] = img[i, :] * float(np.random.normal(1, intensity, 1))
    return img


def set_contrast(img, contrast):
    v_min, v_max = np.percentile(img, contrast)
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))


def fake_spm_image(
    image, noise=0.1, px_size=512, lines=0.02, contrast=(0.1, 98), txt_out=False
):
    """Create fake SPM images by providing input image.

    This is a Fire script file. Execute with the input
    image file and other optional arguments.

    Example:

            $ python fake_spm_images.py input.jpg

    (C) Copyright Nicolas Bock, licensed under GPL v3
    See LICENSE or http://www.gnu.org/licenses/gpl-3.0.html

    Args:
        image (str): Path to image (e.g. jpg, png) file
        noise (float): Optional noise level
        px_size (int): Image size in pixel
        lines (float): Intensity of random line offsets
        contrast (tuple): Lower and higher contrast percentile
        txt_out (bool): Save STM image also to text file

    """

    img = io.imread(image, as_gray=True)
    img_n = add_noise(img, noise)
    img_n_b = blur_image(img_n, px_size)
    img_n_b_l = add_line_distortion(img_n_b, lines)
    img_n_b_l_c = set_contrast(img_n_b_l, contrast)

    fig, ax1 = plt.subplots()
    ax1.imshow(img_n_b_l_c, cmap=plt.get_cmap("YlOrBr"))
    ax1.axis("off")
    fig.savefig("spm_out.png", bbox_inches="tight")

    if txt_out:
        np.savetxt("spm_out.txt", img_n_b_l_c)


if __name__ == "__main__":
    fire.Fire(fake_spm_image)
