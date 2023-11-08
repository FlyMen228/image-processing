import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from math import exp


def image_to_gray(image: Image) -> Image:
    
    return image.convert("L")

def get_image_sizes(image: Image) -> (int, int):
    
    return image.size

def change_contrast(image: Image, width: int, height: int) -> Image:
    
    image_arr = np.array(image)
    
    modified_arr = np.zeros_like(image_arr, dtype=np.float32)
    
    for x in range(height):
        
        for y in range(width):
            
            pixel = image_arr[x, y]
            
            modified_arr[x, y] = 1 / (1 + exp(-(pixel - 0.5)))
            
    return Image.fromarray((modified_arr * 255).astype(np.uint8))

def plot_2_images(image1: Image, image2: Image) -> None:
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("image1")
    plt.imshow(image1, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("image2")
    plt.imshow(image2, cmap="gray")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    original_image = Image.open('original_image.jpg')
    
    gray_image = image_to_gray(original_image)
    
    gray_image.save('gray_image.jpg')
    
    width, height = get_image_sizes(gray_image)
    
    changed_contrast_gray_image = change_contrast(gray_image, width, height)
    
    changed_contrast_gray_image.save('changed_contrast_gray_image.jpg')
    
    plot_2_images(gray_image, changed_contrast_gray_image)