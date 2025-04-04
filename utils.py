import cv2
import numpy as np
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_images(im1, im2):
    h, w, _ = im1.shape
    im1 = np.expand_dims(im1, axis=0)
    im2 = np.expand_dims(im2, axis=0)
    return im1, im2, h, w

def flow_to_image(flow):
    """Visualize optical flow with HSV encoding"""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255                     # Saturation
    hsv[..., 2] = np.clip(magnitude * 255.0, 0, 255)  # Value
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow
