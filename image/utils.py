import cv2
import numpy as np
import tkinter as tk
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt
########################################################################################
def show(img_path):
    window_name = "Image Viewer (OpenCV)"
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((200,200,3), dtype=np.uint8)
        img[:] = (255,255,255)

    # Get screen size (uses Tkinter to get true monitor resolution)
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    h, w = img.shape[:2]

    # compute scale that preserves aspect ratio and keeps image within screen (with margin)
    max_w = int(screen_w * 0.92)
    max_h = int(screen_h * 0.88)
    scale = min(max_w / w, max_h / h, 1.0)  # don't upscale beyond 1.0 by default

    new_w = int(w * scale)
    new_h = int(h * scale)

    # choose interpolation: INTER_AREA for downscaling, INTER_CUBIC for upscaling
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # explicitly set window to resized image size so text layout looks correct
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#############################################################################
def show(img):
    window_name = "Image Viewer (OpenCV)"
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    h, w = img.shape[:2]

    # compute scale that preserves aspect ratio and keeps image within screen (with margin)
    max_w = int(screen_w * 0.92)
    max_h = int(screen_h * 0.88)
    scale = min(max_w / w, max_h / h, 1.0)  # don't upscale beyond 1.0 by default

    new_w = int(w * scale)
    new_h = int(h * scale)

    # choose interpolation: INTER_AREA for downscaling, INTER_CUBIC for upscaling
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # explicitly set window to resized image size so text layout looks correct
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
################################################################################
def remove_noise(img_gray):
    """Remove noise, speckles, and enhance image quality"""
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoising(img_gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive binarization using Sauvola
    window_size = 25
    thresh_s = threshold_sauvola(denoised, window_size=window_size)
    binary = (denoised > thresh_s).astype(np.uint8) * 255
    
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Preserve small components that might be punctuation
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = cleaned.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 10 and h <= 6 and w <= 6:
            result[labels == i] = 255
    
    return result
#####################################################################################
def detect_noise(img_gray, img_color):
    """Detect and mark noise, speckles, and defects"""
    marked_img = img_color.copy()
    
    # 1. Detect black holes/dark defects
    dark_mask = (img_gray < 10).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
    
    # 2. Detect noise speckles using morphological operations
    # Apply median filter
    median = cv2.medianBlur(img_gray, 5)
    diff = cv2.absdiff(img_gray, median)
    _, noise_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 3. Detect small isolated components (salt-and-pepper noise)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = cv2.bitwise_not(binary)
    
    # Find small dark spots
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    small_noise_mask = np.zeros_like(img_gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 < area < 50:  # Small isolated noise
            small_noise_mask[labels == i] = 255
    
    # 4. Combine all defect masks
    combined_mask = cv2.bitwise_or(dark_mask, noise_mask)
    combined_mask = cv2.bitwise_or(combined_mask, small_noise_mask)
    
    # Clean up mask - remove very large regions (likely text)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    final_mask = np.zeros_like(combined_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1000:  # Only mark defects smaller than this
            final_mask[labels == i] = 255
    
    # Mark defects on image with colored boxes and circles
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2:  # Minimum size to mark
            x, y, w, h = cv2.boundingRect(contour)
            # Use different colors for different types of defects
            if area > 100:
                # Larger defects - draw red rectangle
                cv2.rectangle(marked_img, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
            else:
                # Small noise - draw yellow circle
                center = (x + w//2, y + h//2)
                radius = max(3, (w + h) // 4)
                cv2.circle(marked_img, center, radius, (0, 255, 255), 2)
    
    return marked_img, final_mask
################################################################################
def deskew_image(img_gray):
    """Detect and correct image skew"""
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return img_gray, 0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        angles.append(angle)
    median_angle = np.median(angles)
    (h, w) = img_gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle
#############################################################################
def inpaint_defects(img_color, mask_thresh=10):
    """Remove black holes, page folds, and defects"""
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = (gray < mask_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted = cv2.inpaint(img_color, mask, 3, cv2.INPAINT_TELEA)
    return inpainted
#############################################################################
def to_gray(image_path):
    """
        @brief convert RGB inage into gray
        @param image source 
        @return return gray scale image
        @throws not implemented.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_gray
#############################################################################
