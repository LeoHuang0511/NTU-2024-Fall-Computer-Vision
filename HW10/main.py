import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self, image_path):
        self.original_img = np.array(Image.open(image_path)).astype(float)
        
        # Ensure output directory exists
        self.output_dir = 'results'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
    
    def convolution2d(self, img, kernel):
        img_height, img_width = img.shape
        ker_height, ker_width = kernel.shape
        
        pad_h = ker_height // 2
        pad_w = ker_width // 2
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        output = np.zeros_like(img)
        
        for i in range(img_height):
            for j in range(img_width):
                local_region = padded_img[i:i+ker_height, j:j+ker_width]
                
                output[i, j] = np.sum(local_region * kernel)
        
        return output
    
    def binarize(self, img, thr, is_lower=True):
        img_bin = np.zeros_like(img)
        if is_lower:
            img_bin[img < thr] = 255
        else:
            img_bin[img >= thr] = 255
        return img_bin
    
    def laplace_mask1(self):
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        laplace_img = self.convolution2d(self.original_img, kernel)
        return self.binarize(laplace_img, 15)
    
    def laplace_mask2(self):
        kernel = np.array([
            [1., 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ]) / 3
        laplace_img = self.convolution2d(self.original_img, kernel)
        return self.binarize(laplace_img, 15)
    
    def minimum_variance_laplacian(self):
        kernel = np.array([
            [2., -1, 2],
            [-1, -4, -1],
            [2, -1, 2]
        ]) / 3
        laplace_img = self.convolution2d(self.original_img, kernel)
        return self.binarize(laplace_img, 20)
    
    def laplace_of_gaussian(self):
        kernel = np.array([
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
        laplace_img = self.convolution2d(self.original_img, kernel)
        return self.binarize(laplace_img, 3000)
    
    def difference_of_gaussian(self):
        kernel = np.array([
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
        dog_img = self.convolution2d(self.original_img, kernel)
        return self.binarize(dog_img, 1, False)
    
    def process_and_save_images(self):
        methods = [
            ('Original', self.original_img, "None"),
            ('Laplace Mask 1', self.laplace_mask1(), 15),
            ('Laplace Mask 2', self.laplace_mask2(), 15),
            ('Minimum Variance Laplacian', self.minimum_variance_laplacian(), 20),
            ('Laplace Of Gaussian', self.laplace_of_gaussian(), 3000),
            ('Difference Of Gaussian', self.difference_of_gaussian(), 1)
        ]
        
        # Save individual images
        for name, img, _ in methods:
            PIL_image = Image.fromarray(img.astype('uint8'))
            PIL_image.save(os.path.join(self.output_dir, f'{name}.jpg'))
        
        # Create comparison figure
        plt.figure(figsize=(15, 10))
        # plt.suptitle('Edge Detection Methods Comparison', fontsize=16)
        
        for i, (name, img, thre) in enumerate(methods, 1):
            plt.subplot(2, 3, i)
            plt.imshow(img, cmap='gray')
            plt.title(name + f" (Threshold={thre})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison.jpg'))
        plt.close()

# Example usage
if __name__ == '__main__':
    edge_detector = EdgeDetector('lena.bmp')
    edge_detector.process_and_save_images()