"""Create fake SPM images by providing input image.

    (C) Copyright Nicolas Bock, licensed under GPL v3
    See LICENSE or http://www.gnu.org/licenses/gpl-3.0.html
"""

import fire
import numpy as np
from skimage import io, exposure, transform
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


def add_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape)
    return img + noise


def blur_image(img, blur_pixel):
    return transform.downscale_local_mean(img, (blur_pixel, blur_pixel))


def add_line_distortion(img, intensity):
    for i in range(img.shape[0]):
        img[i, :] = img[i, :] * float(np.random.normal(1, intensity, 1))

    return img


def set_contrast(img, low=0.1, high=98):
    v_min, v_max = np.percentile(img, (low, high))
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))


def fake_spm_image(
    image, noise=0.1, blur=5, lines=0.02, contrast=(0.1, 98), txt_out=False
):
    """Create fake SPM images by providing input image.

    This is a Fire script file. Execute with the input
    image file and other optional arguments.

    Example:

            $ python fake_spm_images.py input.jpg

    (C) Copyright Nicolas Bock, licensed under GPL v3
    See LICENSE or http://www.gnu.org/licenses/gpl-3.0.html

    Args:
        image (str): Path to image (e.g. jpg, png) file.
        noise (float): Optional noise level
        blur (int): Number of pixel to be averaged
        lines (float): Intensity of random line offsets
        contrast (tuple): Lower and higher contrast percentile
        txt_out (bool): Save STM image also to text file.


    """
    img = io.imread(image, as_gray=True)
    img_n = add_noise(img, noise_level=noise)
    img_n_b = blur_image(img_n, blur_pixel=blur)
    img_n_b_l = add_line_distortion(img_n_b, intensity=lines)
    img_n_b_l_c = set_contrast(img_n_b_l, low=contrast[0], high=contrast[1])

    fig, ax1 = plt.subplots()
    ax1.imshow(img_n_b_l_c, cmap=plt.get_cmap("YlOrBr"))
    ax1.axis("off")
    scalebar = ScaleBar(
        1.2, "cm", pad=0.6, length_fraction=0.25, frameon=False, location="lower right"
    )
    ax1.add_artist(scalebar)
    fig.savefig("spm_out.png", bbox_inches="tight")

    if txt_out:
        np.savetxt("spm_out.txt", img_n_b_l_c)


if __name__ == "__main__":
    fire.Fire(fake_spm_image)
