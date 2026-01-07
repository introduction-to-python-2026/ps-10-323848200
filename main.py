import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from scipy.signal import convolve2d

def load_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2)
    
    kernelY = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernelX = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    edgeY = convolve2d(gray_image, kernelY, mode='same')
    edgeX = convolve2d(gray_image, kernelX, mode='same')
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG


def main():
    image_path = 'milfe.jpg' 
    original_image = load_image(image_path)
    
    if original_image is None:
        return

    print(f"Image loaded! Shape: {original_image.shape}, Dtype: {original_image.dtype}")


    print("Cleaning noise... (this might take a few seconds)")
    
    clean_image = np.zeros_like(original_image)
    for i in range(3):
        clean_image[:, :, i] = median(original_image[:, :, i], disk(3))

    print("Detecting edges...")
    edge_mag = edge_detection(clean_image)

    threshold_value = 50 
    binary_edges = edge_mag > threshold_value

    edge_image_to_save = Image.fromarray((binary_edges * 255).astype(np.uint8))
    edge_image_to_save.save('my_edges.png')
    
    print("Process completed. Result saved as 'my_edges.png'")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original (milfe.jpg)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary_edges, cmap='gray')
    plt.title(f"Edges (Threshold={threshold_value})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
