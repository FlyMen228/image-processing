import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.fft import fft2, ifft2
from matplotlib.patches import Rectangle


def image_to_gray(image: Image) -> Image:
    
    return image.convert("L")


def get_image_sizes(image: Image) -> (int, int):
    
    return image.size


def normalize_image(image: Image) -> list:
    
    return np.array(image) / 255

def normalized_arr_to_image(image_arr: list) -> Image:
    
    return Image.fromarray((image_arr * 255).astype(np.uint8))


def extend_images(image_arr: list, fragment_arr: list, height: int, width: int, fragment_height: int, fragment_width: int) -> (list, list):
    
    extended_width, extended_height = width + fragment_width - 1, height + fragment_height - 1

    temp_arr = np.zeros((height, width))

    center_mass = np.unravel_index(np.argmax(image_arr), image_arr.shape)
    i, j = center_mass

    i_start, i_end = max(0, i - fragment_width // 2), min(width, i + fragment_width // 2)
    j_start, j_end = max(0, j - fragment_height // 2), min(height, j + fragment_height // 2)
    
    temp_arr[i_start:i_end, j_start:j_end] = fragment_arr

    extended_image_arr = np.zeros((extended_height, extended_width))
    extended_fragment_arr = np.zeros((extended_height, extended_width))

    extended_image_arr[:height, :width] = image_arr

    extended_fragment_arr[:height, :width] = temp_arr

    return extended_image_arr, extended_fragment_arr


def linear_correlation(image_arr: list, fragment_arr: list) -> list:
    
    Fu = fft2(image_arr)
    
    Fv_conj = np.conj(fft2(fragment_arr))
    
    Fu_x_Fv_conj = Fu * Fv_conj
    
    width, height = image_arr.shape
    fragment_width, fragment_height = fragment_arr.shape
    
    P, R = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width))
    
    exp = np.exp(-2j * np.pi * (P * (fragment_width - 1) / width + R * (fragment_height - 1) / height))
    
    return np.abs(ifft2(exp * Fu_x_Fv_conj))


def reduce_matrix(linear_cor_arr: list, width: int, height: int, fragment_width: int, fragment_height: int) -> (int, int, int, int):
    
    max_index = np.unravel_index(np.argmax(linear_cor_arr), linear_cor_arr.shape)
    
    i_start = max(0, max_index[0] - fragment_height // 2)
    i_end = min(height, max_index[0] + fragment_height // 2 + 1)

    j_start = max(0, max_index[1] - fragment_width // 2)
    j_end = min(width, max_index[1] + fragment_width // 2 + 1)
    
    return (i_start, i_end, j_start, j_end)


def plot_3_images_w_localized_fragment(image1: Image, image2: Image, cords: tuple) -> None:
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image1)
    ax[0].set_title("Оригинальное изображение")
    ax[0].axis('off')

    ax[1].imshow(image2)
    ax[1].set_title("Фрагмент")
    ax[1].axis('off')

    rectangle = Rectangle((cords[3], cords[0]), cords[3] - cords[2], cords[1] - cords[0], linewidth=1, edgecolor='r', facecolor='none')
    
    ax[2].imshow(image1)
    ax[2].set_title("Оригинальное изображение с локализованным фрагментом")
    ax[2].add_patch(rectangle)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    
    original_image = Image.open('original_image.jpg')
    
    gray_image = image_to_gray(original_image)
    
    width, height = get_image_sizes(gray_image)
    
    gray_image_array = normalize_image(gray_image)
    
    fragment = Image.open('fragment.jpg')
    
    fragment_gray = image_to_gray(fragment)
    
    fragment_arr = normalize_image(fragment_gray)
    
    fragment_height, fragment_width = get_image_sizes(fragment_gray)
    
    extended_gray_image_arr, extended_fragment_arr = extend_images(gray_image_array, fragment_arr, height, width, fragment_height, fragment_width)
    
    linear_correlation_arr = linear_correlation(extended_gray_image_arr, extended_fragment_arr)

    reduced_cords = reduce_matrix(linear_correlation_arr, width, height, fragment_width, fragment_height)
    
    plot_3_images_w_localized_fragment(original_image, fragment, reduced_cords)

if __name__ == '__main__':
    
    main()