import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import convolve2d


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def image_to_gray(image: Image) -> Image:
    
    return image.convert("L")

def get_image_sizes(image: Image) -> (int, int):
    
    return image.size

def normalize_image(image: Image) -> list:
    
    return np.array(image) / 255

def operator_sobel(image_array: list) -> Image:
    
    gradient_x = convolve2d(image_array, sobel_x, mode='same', boundary='symm', fillvalue=0)
    gradient_y = convolve2d(image_array, sobel_y, mode='same', boundary='symm', fillvalue=0)
    
    return Image.fromarray((np.sqrt(gradient_x**2 + gradient_y**2) * 255).astype(np.uint8))

def L2_gradient(image_arr: list) -> Image:
    
    gradient_x = np.diff(image_arr, axis=1, prepend=0)
    gradient_y = np.diff(image_arr, axis=0, prepend=0)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return Image.fromarray((gradient_magnitude * 255).astype(np.uint8))

def L1_gradient(image_arr: list) -> Image:
    
    gradient_x = np.abs(np.diff(image_arr, axis=1, prepend=0))
    gradient_y = np.abs(np.diff(image_arr, axis=0, prepend=0))
    
    gradient_magnitude = gradient_x + gradient_y
    
    return Image.fromarray((gradient_magnitude * 255).astype(np.uint8))

def plot_4_images(image1: Image, image2: Image, image3: Image, image4: Image) -> None:
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("gray")
    plt.imshow(image1, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("sobel")
    plt.imshow(image2, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("L2")
    plt.imshow(image3, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("L1")
    plt.imshow(image4, cmap="gray")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    original_image = Image.open('original_image.jpg')
    
    gray_image = image_to_gray(original_image)
    
    gray_image.save('gray_image.jpg')
    
    width, height = get_image_sizes(gray_image)
    
    gray_image_array = normalize_image(gray_image)
    
    sobel_image = operator_sobel(gray_image_array)
    
    sobel_image.save('sobel_image.jpg')
    
    L2_image = L2_gradient(gray_image_array)
    
    L2_image.save('L2_image.jpg')
    
    L1_image = L1_gradient(gray_image_array)
    
    L1_image.save('L1_image.jpg')
    
    plot_4_images(gray_image, sobel_image, L2_image, L1_image)