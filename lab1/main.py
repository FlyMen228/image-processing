from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def convert_to_gray(original_image: Image) -> Image:
    
    return original_image.convert('L')


def calc_intesity_counts(gray_image: Image) -> list:
    
    gray_arr = np.array(gray_image)
    
    intensity_counts = np.zeros(256, dtype=int)
    
    for intensity in range(256):
        
        intensity_counts[intensity] = np.sum(gray_arr == intensity)
        
    return intensity_counts


def show_2(image: Image, intensity_counts: list) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Серое изображение')
    axes[0].axis('off')
    
    first_non_zero_intensity = np.argmax(intensity_counts != 0)
    last_non_zero_intensity = 255 - np.argmax(np.flip(intensity_counts) != 0)
    
    plt.bar(range(first_non_zero_intensity, last_non_zero_intensity + 1), 
            intensity_counts[first_non_zero_intensity:last_non_zero_intensity + 1])
    axes[1].set_title('Гистограмма интенсивностей')
    axes[1].set_xlabel('Интенсивность')
    axes[1].set_ylabel('Количество пикселей')
    
    plt.tight_layout()
    plt.show()


def main():
    
    original_image = Image.open('./original_image.jpg')
    
    gray_image = convert_to_gray(original_image)
    
    intensity_counts = calc_intesity_counts(gray_image)
    
    show_2(gray_image, intensity_counts)


if __name__ == '__main__':
    
    main()