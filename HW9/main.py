import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class EdgeDetector:
    def __init__(self, image_path):
        self.original_image = Image.open(image_path).convert('L')
        self.image_array = np.array(self.original_image, dtype=np.float32)
        self.width, self.height = self.original_image.size
    
    def _expand_image(self, arr):
        return np.pad(arr, pad_width=1, mode='edge')

    def roberts_operator(self, threshold=12):
        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        
        for i in range(self.height):
            for j in range(self.width):
                gx = expanded[i+1, j+1] - expanded[i+2, j+2]
                gy = expanded[i+2, j+1] - expanded[i+1, j+2]
                grad = np.sqrt(gx**2 + gy**2)
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def prewitt_operator(self, threshold=24):
        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        
        for i in range(self.height):
            for j in range(self.width):
                gx = np.sum(expanded[i+2, j:j+3]) - np.sum(expanded[i, j:j+3])
                gy = np.sum(expanded[i:i+3, j+2]) - np.sum(expanded[i:i+3, j])
                grad = np.sqrt(gx**2 + gy**2)
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def sobel_operator(self, threshold=38):
        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        
        for i in range(self.height):
            for j in range(self.width):
                gx = (
                    expanded[i+2, j] + 2*expanded[i+2, j+1] + expanded[i+2, j+2] -
                    (expanded[i, j] + 2*expanded[i, j+1] + expanded[i, j+2])
                )
                
                gy = (
                    expanded[i, j+2] + 2*expanded[i+1, j+2] + expanded[i+2, j+2] -
                    (expanded[i, j] + 2*expanded[i+1, j] + expanded[i+2, j])
                )
                
                grad = np.sqrt(gx**2 + gy**2)
                
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def frei_chen_operator(self, threshold=30):
        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        sqrt2 = np.sqrt(2)
        
        for i in range(self.height):
            for j in range(self.width):
                gx = (
                    expanded[i+2, j] + sqrt2*expanded[i+2, j+1] + expanded[i+2, j+2] -
                    (expanded[i, j] + sqrt2*expanded[i, j+1] + expanded[i, j+2])
                )
                
                gy = (
                    expanded[i, j+2] + sqrt2*expanded[i+1, j+2] + expanded[i+2, j+2] -
                    (expanded[i, j] + sqrt2*expanded[i+1, j] + expanded[i+2, j])
                )
                
                grad = np.sqrt(gx**2 + gy**2)
                
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def kirsch_operator(self, threshold=135):
    
        def kernel_operation(array, pos, neg):
            p = 0
            for i in pos:
                p += 5*array[i[0]][i[1]]
            for i in neg:
                p -= 3*array[i[0]][i[1]]
            
            return p

        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        
        for i in range(self.height):
            for j in range(self.width):
                p1 = kernel_operation(expanded, ((i, j+2), (i+1, j+2), (i+2, j+2)), ((i, j), (i, j+1), (i+1, j), (i+2, j), (i+2, j+1)))
                p2 = kernel_operation(expanded, ((i, j), (i+1, j), (i+2, j)), ((i, j+2), (i, j+1), (i+1, j+2), (i+2, j+2), (i+2, j+1)))
                p3 = kernel_operation(expanded, ((i, j), (i, j+1), (i, j+2)), ((i+1, j), (i+2, j), (i+1, j+2), (i+2, j+2), (i+2, j+1)))
                p4 = kernel_operation(expanded, ((i+2, j), (i+2, j+1), (i+2, j+2)), ((i, j), (i+1, j), (i, j+2), (i+1, j+2), (i, j+1)))
                p5 = kernel_operation(expanded, ((i, j), (i+1, j), (i, j+1)), ((i, j+2), (i+1, j+2), (i+2, j+2), (i+2, j+1), (i+2, j)))
                p6 = kernel_operation(expanded, ((i, j+2), (i+1, j+2), (i, j+1)), ((i, j), (i+1, j), (i+2, j), (i+2, j+1), (i+2, j+2)))
                p7 = kernel_operation(expanded, ((i+2, j+2), (i+1, j+2), (i+2, j+1)), ((i, j), (i, j+1), (i+1, j), (i+2, j), (i, j+2)))
                p8 = kernel_operation(expanded, ((i+2, j), (i+1, j), (i+2, j+1)), ((i, j), (i, j+1), (i, j+2), (i+1, j+2), (i+2, j+2)))
                
                grad = max(p1, p2, p3, p4, p5, p6, p7, p8)
                
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def robinson_operator(self, threshold=43):
        expanded = self._expand_image(self.image_array)
        result = np.zeros_like(self.image_array)
        
        for i in range(self.height):
            for j in range(self.width):
                p1 = abs(
                    expanded[i][j+2] + 2*expanded[i+1][j+2] + expanded[i+2][j+2] -
                    (expanded[i][j] + 2*expanded[i+1][j] + expanded[i+2][j])
                )
                
                p3 = abs(
                    expanded[i][j] + 2*expanded[i][j+1] + expanded[i][j+2] -
                    (expanded[i+2][j] + expanded[i+2][j+2] + 2*expanded[i+2][j+1])
                )
                
                p5 = abs(
                    2*expanded[i][j] + expanded[i+1][j] + expanded[i][j+1] -
                    (expanded[i+1][j+2] + 2*expanded[i+2][j+2] + expanded[i+2][j+1])
                )
                
                p6 = abs(
                    2*expanded[i][j+2] + expanded[i+1][j+2] + expanded[i][j+1] -
                    (expanded[i+1][j] + 2*expanded[i+2][j] + expanded[i+2][j+1])
                )
                
                grad = max(p1, p3, p5, p6)
                
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def nevatia_babu_operator(self, threshold=12500):
        expanded = self._expand_image(self.image_array)
        expanded = self._expand_image(expanded)

        result = np.zeros_like(self.image_array)

        for i in range(self.height):
            for j in range(self.width):
                p1 = (
                    100 * sum(expanded[i][j+k] for k in range(5)) +
                    100 * sum(expanded[i+1][j+k] for k in range(5)) -
                    100 * sum(expanded[i+3][j+k] for k in range(5)) -
                    100 * sum(expanded[i+4][j+k] for k in range(5))
                )
                
                p2 = (
                    100 * sum(expanded[i+k][j+4] for k in range(5)) +
                    100 * sum(expanded[i+k][j+3] for k in range(5)) -
                    100 * sum(expanded[i+k][j+1] for k in range(5)) -
                    100 * sum(expanded[i+k][j] for k in range(5))
                )
                
                p3 = (
                    100 * sum(expanded[i][j+k] for k in range(5)) +
                    100 * sum(expanded[i+1][j+k] for k in range(3)) +
                    100 * expanded[i+2][j] +
                    78 * expanded[i+1][j+3] +
                    92 * expanded[i+2][j+1] +
                    32 * expanded[i+3][j] -
                    100 * sum(expanded[i+4][j+k] for k in range(5)) -
                    100 * sum(expanded[i+3, j+2:j+5]) -
                    100 * expanded[i+2][j+4] -
                    78 * expanded[i+3][j+1] -
                    92 * expanded[i+2][j+3] -
                    32 * expanded[i+1][j+4]
                )
                
                p4 = (
                    100 * sum(expanded[i+k][j] for k in range(5)) +
                    100 * (expanded[i][j+1] + expanded[i+1][j+1] + expanded[i+1][j+2] +  expanded[i+2][j+2]) +
                    78 * expanded[i+3][j+1] +
                    92 * expanded[i+1][j+2] +
                    32 * expanded[i][j+3] -
                    100 * sum(expanded[i+k][j+4] for k in range(5)) -
                    100 * sum(expanded[i+2:i+5,j+3]) -
                    100 * expanded[i+4][j+2] -
                    78 * expanded[i+1][j+3] -
                    92 * expanded[i+3][j+2] -
                    32 * expanded[i+4][j+1]
                )
                
                p5 = (
                    100 * sum(expanded[i+k][j+4] for k in range(5)) +
                    100 * sum(expanded[i:i+3, j+3]) +
                    100 * expanded[i][j+2] + 
                    78 * expanded[i+3][j+3] +
                    92 * expanded[i+1][j+2] +
                    32 * expanded[i][j+1] -
                    100 * sum(expanded[i+k][j] for k in range(5)) -
                    100 * sum(expanded[i+2:i+5, j+1]) -
                    100 * expanded[i+4, j+2] -
                    78 * expanded[i+1][j+1] -
                    92 * expanded[i+3][j+2] -
                    32 * expanded[i+4][j+3]
                )
                
                p6 = (
                    100 * sum(expanded[i][j+k] for k in range(5)) +
                    100 * sum(expanded[i+1, j+2:j+5]) +
                    100 * expanded[i+2][j+4] +
                    78 * expanded[i+1][j+1] +
                    92 * expanded[i+2][j+3] +
                    32 * expanded[i+3][j+4] -
                    100 * sum(expanded[i+4][j+k] for k in range(5)) -
                    100 * sum(expanded[i+3, j:j+3]) -
                    100 * expanded[i+2][j] -
                    78 * expanded[i+3][j+3] -
                    92 * expanded[i+2][j+1] -
                    32 * expanded[i+1][j]
                )
                
                grad = max(p1, p2, p3, p4, p5, p6)
                
                result[i, j] = 0 if grad >= threshold else 255
        
        return result

    def save_result(self, result, method_name):
        output_image = Image.fromarray(result.astype(np.uint8))
        
        os.makedirs('results', exist_ok=True)
        output_path = os.path.join('results', f'{method_name}_result.png')
        output_image.save(output_path)
        print(f"Saved result for {method_name} at {output_path}")
    
    def save_comparative_figure(self, methods=None):
        if methods is None:
            methods = [
                ('Roberts', self.roberts_operator, 12),
                ('Prewitt', self.prewitt_operator, 24),
                ('Sobel', self.sobel_operator, 38),
                ('Frei-Chen', self.frei_chen_operator, 30),
                ('Kirsch', self.kirsch_operator, 135),
                ('Robinson', self.robinson_operator, 43),
                ('Nevatia-Babu', self.nevatia_babu_operator, 12500)
            ]
        
        num_methods = len(methods)
        cols = (num_methods + 1) // 2  
        rows = 2 if num_methods > 1 else 1
        
        plt.figure(figsize=(16, 4 * rows))
        
        plt.subplot(rows, cols, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        for i, (method_name, method_func, threshold) in enumerate(methods, start=2):
            result = method_func(threshold)
            
            plt.subplot(rows, cols, i)
            plt.imshow(result, cmap='gray')
            plt.title(f'{method_name} (Threshold: {threshold})')
            plt.axis('off')
        
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        
        output_path = os.path.join('results', 'edge_detection_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparative edge detection figure at {output_path}")

def main():
    input_image_path = './lena.bmp' 
    edge_detector = EdgeDetector(input_image_path)
    
    methods = [
        ('roberts', edge_detector.roberts_operator),
        ('prewitt', edge_detector.prewitt_operator),
        ('sobel', edge_detector.sobel_operator),
        ('frei_chen', edge_detector.frei_chen_operator),
        ('kirsch', edge_detector.kirsch_operator),
        ('robinson', edge_detector.robinson_operator),
        ('nevatia_babu', edge_detector.nevatia_babu_operator)
    ]
    
    for method_name, method in methods:
        result = method()
        edge_detector.save_result(result, method_name)
    
    edge_detector.save_comparative_figure()

if __name__ == "__main__":
    main()



