import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """
    Loads an image from a file path and converts it to a NumPy array.
    """
    # פתיחת התמונה באמצעות PIL
    img = Image.open(path)
    # המרה למערך NumPy והחזרה
    return np.array(img)

def edge_detection(image):
    """
    Performs edge detection using Sobel filters.
    """
    # 1. Convert to Grayscale
    # המרה לגווני אפור על ידי ממוצע של שלושת ערוצי הצבע (R, G, B)
    # image shape is usually (Height, Width, 3) -> we calculate mean on axis 2
    gray_image = np.mean(image, axis=2)

    # 2. Define Sobel Kernels (Based on lecture slide 61)
    # זיהוי שינויים אנכיים (קצוות אופקיים)
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    
    # זיהוי שינויים אופקיים (קצוות אנכיים)
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # 3. Apply Convolution (Based on lecture slides 20-25, 38-42)
    # mode='same' מבטיח שהפלט יהיה באותו גודל כמו הקלט
    # boundary='fill' ו-fillvalue=0 מטפלים בקצוות התמונה (Padding)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)

    # 4. Compute Magnitude (Based on lecture slide 59)
    # שילוב שני הכיוונים לקבלת עוצמת הקצה הכללית
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
