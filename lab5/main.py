import numpy as np

from PIL import Image
from scipy.fft import fft2, ifft2


def image_to_gray(image: Image) -> Image:
    
    return image.convert("L")

def get_image_sizes(image: Image) -> (int, int):
    
    return image.size

def input_fragment_size(width: int, height: int) -> (int, int):
    
    input_string = input('Введите ширину и высоту фрагмента изображения через пробел: ').split()
    
    return (int(input_string[0]) if width >= int(input_string[0]) else width, 
            int(input_string[1]) if height >= int(input_string[1]) else height)

def normalize_image(image: Image) -> list:
    
    return np.array(image) / 255

def extract_fragment(image: list, fragment_width: int, fragment_height: int) -> list:
    
    center_mass = np.unravel_index(np.argmax(image), image.shape)
    i, j = center_mass
    
    i_start, i_end = max(0, i - fragment_height // 2), min(image.shape[0], i + fragment_height // 2)
    j_start, j_end = max(0, j - fragment_width // 2), min(image.shape[1], j + fragment_width // 2)
    
    return image[i_start:i_end, j_start:j_end]

def normalized_arr_to_image(image_arr: list) -> Image:
    
    return Image.fromarray((image_arr * 255).astype(np.uint8))

def extend_images(image_arr: list, fragment_arr: list, height: int, width: int, fragment_height: int, fragment_width: int) -> (list, list):
    
    extended_width, extended_height = width + fragment_width - 1, height + fragment_height - 1

    temp_arr = np.zeros((height, width))

    center_mass = np.unravel_index(np.argmax(image_arr), image_arr.shape)
    i, j = center_mass

    i_start, i_end = max(0, i - fragment_height // 2), min(height, i + fragment_height // 2)
    j_start, j_end = max(0, j - fragment_width // 2), min(width, j + fragment_width // 2)
    
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
    
    image_x_fragment = ifft2(Fu_x_Fv_conj)
    
    return np.abs(image_x_fragment)

def inverse_linear_correlation(linear_correlation_arr: list) -> list:
    
    return np.abs(ifft2(linear_correlation_arr))

def reduce_matrix(arr: list, height: int, width: int, fragment_height: int, fragment_width: int) -> list:
    
    center_mass = np.unravel_index(np.argmax(arr), arr.shape)
    i, j = center_mass

    i_start, i_end = max(0, i - height // 2), min(fragment_height, i + height // 2)
    j_start, j_end = max(0, j - width // 2), min(fragment_width, j + width // 2)

    return arr[i_start : i_end, j_start : j_end]

def find_max_elem(arr: list) -> int:
    
    return np.max(arr)


def main():
    
    original_image = Image.open('original_image.jpg')
    
    gray_image = image_to_gray(original_image)
    
    gray_image.save('gray_image.jpg')
    
    width, height = get_image_sizes(gray_image)
    
    fragment_width, fragment_height = input_fragment_size(width, height)
    
    gray_image_array = normalize_image(gray_image)
    
    fragment_arr = extract_fragment(gray_image_array, fragment_width, fragment_height)
    
    normalized_arr_to_image(fragment_arr).save('fragment.jpg')
    
    extended_gray_image_arr, extended_fragment_arr = extend_images(gray_image_array, fragment_arr, height, width, fragment_height, fragment_width)
    
    normalized_arr_to_image(extended_gray_image_arr).save('extended_gray_image.jpg')
    
    normalized_arr_to_image(extended_fragment_arr).save('extended_fragment.jpg')
    
    linear_correlation_arr = linear_correlation(extended_gray_image_arr, extended_fragment_arr)
    normalized_arr_to_image(linear_correlation_arr).save('linear_correlation.jpg')
    
    inverse_linear_correlation_arr = inverse_linear_correlation(linear_correlation_arr)
    normalized_arr_to_image(inverse_linear_correlation_arr).save('inverse_linear_correlation.jpg')

    reduced_matrix = reduce_matrix(inverse_linear_correlation_arr, height, width, fragment_height, fragment_width)
    normalized_arr_to_image(reduced_matrix).save('reduced_image.jpg')
    
    max_elem = find_max_elem(reduced_matrix)
    
    print(f'Максимальный элемент в редуцированной матрице: {max_elem}')
    

if __name__ == '__main__':
    main()