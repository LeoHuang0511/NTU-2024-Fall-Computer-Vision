import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class NoiseProcessor:
    def __init__(self, image_path):
        self.image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
        self.output_dir = 'processed_results'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_noise(self, noise_type='gaussian', params=None):
        if noise_type == 'gaussian':
            amplitude = params.get('amplitude', 10)
            noise = amplitude * np.random.normal(0, 1, self.image.shape)
            return (self.image.astype(float) + noise).astype(np.uint8)
        
        elif noise_type == 'salt_pepper':
            threshold = params.get('threshold', 0.1)
            noisy = self.image.copy()
            mask = np.random.uniform(0, 1, self.image.shape)
            h, w = self.image.shape
            for i in range(h):
                for j in range(w):
                    if mask[i, j] < threshold:
                        noisy[i, j] = 0
                    elif mask[i, j] > 1 - threshold: 
                        noisy[i, j] = 255
            
            return noisy

    def filtering(self, image, filter_type='box', kernel_size=3):
        if filter_type == 'box':
            return self._box_filter(image, kernel_size)
        elif filter_type == 'median':
            return self._median_filter(image, kernel_size)
        
    def padding(self, image, pad_size, mode='edge'):
        height, width = image.shape
        padded = np.zeros((height + 2*pad_size, width + 2*pad_size), dtype=image.dtype)
        
        # Copy original image
        padded[pad_size:pad_size+height, pad_size:pad_size+width] = image

        # Handle edge padding
        if mode == 'edge':
            # Top and bottom padding
            for y in range(pad_size):
                padded[y, pad_size:pad_size+width] = image[0, :]
                padded[height+pad_size+y, pad_size:pad_size+width] = image[-1, :]
            
            # Left and right padding
            for x in range(pad_size):
                padded[:, x] = padded[:, pad_size]
                padded[:, width+pad_size+x] = padded[:, pad_size+width-1]
        
        return padded
    
    def median(self, values):
        sorted_values = sorted(values.reshape(-1))
        length = len(sorted_values)
        mid = length // 2
        
        if length % 2 == 0:
            return (sorted_values[mid-1] + sorted_values[mid]) / 2
        else:
            return sorted_values[mid]

    def _box_filter(self, image, kernel_size):
        h, w = image.shape
        output = np.zeros_like(image)
        pad = kernel_size // 2
        padded = self.padding(image, pad, mode='edge')
        
        for i in range(h):
            for j in range(w):
                output[i,j] = padded[i:i+kernel_size, j:j+kernel_size].mean()
        
        return output.astype(np.uint8)

    def _median_filter(self, image, kernel_size):
        h, w = image.shape
        output = np.zeros_like(image)
        pad = kernel_size // 2
        padded = self.padding(image, pad, mode='edge')
        
        for i in range(h):
            for j in range(w):
                output[i,j] = self.median(padded[i:i+kernel_size, j:j+kernel_size])
        
        return output.astype(np.uint8)

    def calculate_snr(self, original, noisy):
        noise = noisy.astype(float) - original.astype(float)
        return np.log10(original.var()/noise.var()) * 10

    def morphological_transform(self, image, transform_type='open_close'):
        kernel = np.array([[-2, -1], [-2, 0], [-2, 1],
                   [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
                   [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                   [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                   [2, -1], [2, 0], [2, 1]])
        
        def opening(img, kernel):
            return dilation(erosion(img, kernel), kernel)
        def closing(img, kernel):
            return erosion(dilation(img, kernel), kernel)
        
        if transform_type == 'open_close':
            opened = opening(image, kernel)
            return closing(opened, kernel).astype(np.uint8)
        else:
            closed = closing(image, kernel)
            return opening(closed, kernel).astype(np.uint8)

    def process_and_save(self):
        # (a) Generate Gaussian Noise
        gaussian_noises = {}
        print("Processing Gaussian noise")
        for amp in [10, 30]:
            noise = self.generate_noise('gaussian', {'amplitude': amp})
            gaussian_noises[amp] = noise
            self._save_single_image(noise, f'gaussian_noise_amp_{amp}')
        self._save_comparison_figure(gaussian_noises, 'gaussian_noise')

        # (b) Generate Salt-and-Pepper Noise
        salt_pepper_noises = {}
        print("Processing salt and pepper noise")
        for prob in [0.1, 0.05]:
            noise = self.generate_noise('salt_pepper', {'threshold': prob})
            salt_pepper_noises[prob] = noise
            self._save_single_image(noise, f'salt_pepper_noise_prob_{prob}')
        self._save_comparison_figure(salt_pepper_noises, 'salt_pepper_noise')

        # (c) Box Filtering
        box_filtered_images = {}
        for noise_type, noises in [('gaussian', gaussian_noises), ('salt_pepper', salt_pepper_noises)]:
            print(f"Processing box filtering noise of {noise_type}")
            for param, noisy_image in noises.items():
                for kernel in [3, 5]:
                    filtered = self.filtering(noisy_image, 'box', kernel)
                    key = f'{noise_type}_{param}_box_{kernel}'
                    box_filtered_images[key] = filtered
                    self._save_single_image(filtered, key)
        self._save_comparison_figure(box_filtered_images, 'box_filtering')

        # (d) Median Filtering
        median_filtered_images = {}
        for noise_type, noises in [('gaussian', gaussian_noises), ('salt_pepper', salt_pepper_noises)]:
            print(f"Processing median filtering noise of {noise_type}")
            for param, noisy_image in noises.items():
                for kernel in [3, 5]:
                    filtered = self.filtering(noisy_image, 'median', kernel)
                    key = f'{noise_type}_{param}_median_{kernel}'
                    median_filtered_images[key] = filtered
                    self._save_single_image(filtered, key)
        self._save_comparison_figure(median_filtered_images, 'median_filtering')

        # (e) Morphological Transforms
        morphological_images = {}
        for noise_type, noises in [('gaussian', gaussian_noises), ('salt_pepper', salt_pepper_noises)]:
            print(f"Processing morphological transform noise of {noise_type}")
            for param, noisy_image in noises.items():
                for transform_type in ['open_close', 'close_open']:
                    filtered = self.morphological_transform(noisy_image, transform_type)
                    key = f'{noise_type}_{param}_{transform_type}'
                    morphological_images[key] = filtered
                    self._save_single_image(filtered, key)
        self._save_comparison_figure(morphological_images, 'morphological_transforms')

    def _save_single_image(self, image, filename):
        snr_value = self.calculate_snr(self.image, image)
        filepath = os.path.join(self.output_dir, f'{filename}_snr_{snr_value:.4f}.jpg')
        Image.fromarray(image).save(filepath)

    def _save_comparison_figure(self, images, title):
        plt.figure(figsize=(15, 10))
        
        cols = min(4, len(images))
        rows = (len(images) + cols - 1) // cols
        
        for i, (key, image) in enumerate(images.items(), 1):
            plt.subplot(rows, cols, i)
            plt.imshow(image, cmap='gray')
            snr = self.calculate_snr(self.image, image)
            plt.title(f'{key}\nSNR: {snr:.2f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(title.replace('_', ' ').title(), fontsize=16)
        plt.savefig(os.path.join(self.output_dir, f'{title}_comparison.png'), dpi=300)
        plt.close()

def dilation(img, kernel):
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                maximum = 0
                idx = []
                jdx = []
                for (k, l) in kernel:
                    i_d = i + k
                    j_d = j + l
                    if 0 <= i_d < img.shape[0] and 0 <= j_d < img.shape[1]:
                        maximum = max(maximum, img[i_d][j_d])
                        idx.append(i_d)
                        jdx.append(j_d)
                img_temp[(idx, jdx)] = maximum
    
    return img_temp

def erosion(img, kernel):
    img_temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            draw = True
            minimum = 255
            for (k, l) in kernel:
                i_d = i + k
                j_d = j + l
                if i_d < 0 or j_d < 0 or i_d >= img.shape[0] or j_d >= img.shape[1] or img[i_d][j_d] <= 0:
                    draw = False
                    break
                minimum = min(minimum, img[i_d][j_d])
            if draw:
                img_temp[i, j] = minimum
    
    return img_temp

def main():
    processor = NoiseProcessor('lena.bmp')
    processor.process_and_save()

if __name__ == "__main__":
    main()