import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from math import log10, sqrt


def image_to_gray(image: Image) -> Image:
    
    return image.convert("L")

def get_image_sizes(image: Image) -> (int, int):
    
    return image.size

def add_noise_to_image_zero_squared(image: Image, sigma: int) -> Image:

    image_arr = np.array(image)

    noise = np.random.normal(0, sigma**2, image_arr.shape).astype(np.uint8)
    
    noise_image_arr = np.add(image_arr, noise)
    
    noise_image = Image.fromarray(noise_image_arr)
    
    return noise_image

def calc_with_smooth_window_mean(image: Image, width: int, height: int, n: int) -> Image:
    
    image_arr = np.array(image)
    
    image3_arr = np.zeros_like(image_arr, dtype=np.uint8)
    
    for y in range(n, height - n):
        
        for x in range(n, width - n):
            
            window = image_arr[y - n : y + n + 1, x - n : x + n + 1]

            image3_arr[y, x] = np.mean(window)
            
    image3 = Image.fromarray(image3_arr)
    
    return image3

def calc_with_smooth_window_median(image: Image, width: int, height: int, n: int) -> Image:
    
    image_arr = np.array(image)
    
    image4_arr = np.zeros_like(image_arr, dtype=np.uint8)
    
    for y in range(n, height - n):
        
        for x in range(n, width - n):
            
            window = image_arr[y - n : y + n + 1, x - n : x + n + 1]

            image4_arr[y, x] = np.median(window)
            
    image4 = Image.fromarray(image4_arr)
    
    return image4

def plot_4_images(image1: Image, image2: Image, image3: Image, image4: Image) -> None:
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("image1")
    plt.imshow(image1, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("image2")
    plt.imshow(image2, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("image3")
    plt.imshow(image3, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("image4")
    plt.imshow(image4, cmap="gray")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_mse(original_image_arr: list[float], noise_image_arr: list[float], width: int, height: int) -> float:
    
    mse = 0
    
    for i in range(height):
        
        for j in range(width):
            
            mse += (original_image_arr[i, j] - noise_image_arr[i, j])**2
            
    return mse / (height * width)

def calculate_psnr(original_image: Image, noise_image: Image, width: int, height: int) -> float:
    
    original_image_arr = np.array(original_image, dtype=np.float32)
    noise_image_arr = np.array(noise_image, dtype=np.float32)
    
    mse = calculate_mse(original_image_arr, noise_image_arr, width, height)
    
    MAX = 255
    
    return 20 * log10(MAX / sqrt(mse))


def main():
    
    original_image = Image.open('original_image.jpg')
    
    gray_image = image_to_gray(original_image)
    
    gray_image.save("gray_image.jpg")
    
    width, height = get_image_sizes(gray_image)
    
    noise_image = add_noise_to_image_zero_squared(gray_image, 5)
    
    noise_image.save("noise_image.jpg")
    
    image3 = calc_with_smooth_window_mean(noise_image, width, height, 5)
    
    image3.save("image3.jpg")
    
    image4 = calc_with_smooth_window_median(noise_image, width, height, 5)
    
    image4.save("image4.jpg")
    
    plot_4_images(gray_image, noise_image, image3, image4)
    
    print(f"PSNR(noise_image.jpg, image3.jpg) = {calculate_psnr(noise_image, image3, width, height)} dB")
    print(f"PSNR(noise_image.jpg, image4.jpg) = {calculate_psnr(noise_image, image4, width, height)} dB")


if __name__ == '__main__':
    
    main()