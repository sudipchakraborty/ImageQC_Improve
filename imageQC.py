# Requires: pip install opencv-python scikit-image numpy matplotlib
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt

import filter as fltr
import image as img


"""Module for creating visual comparisons"""
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

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

def detect_noise_and_defects(img_gray, img_color):
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

def remove_noise_and_enhance(img_gray):
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

def inpaint_defects(img_color, mask_thresh=10):
    """Remove black holes, page folds, and defects"""
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = (gray < mask_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted = cv2.inpaint(img_color, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def process_and_compare(image_path, output_path="comparison.png"):
    """Process image and create side-by-side comparison with marked defects"""
    # # Read image
    # img = cv2.imread(image_path)
    # if img is None:
    #     print(f"Error: Could not read image from {image_path}")
    #     return
    
    # # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # Detect and mark defects on original image
    # marked_original, defect_mask = detect_noise_and_defects(gray, img)
    # num_defects = cv2.countNonZero(defect_mask)
    # print(f"Detected defects: {num_defects} pixels marked")
    
    # # Step 1: Deskew
    # deskewed, angle = deskew_image(gray)
    # print(f"Detected skew angle: {angle:.2f} degrees")
    
    # # Rotate original color image by same angle
    # (h, w) = img.shape[:2]
    # M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    # img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # marked_rotated = cv2.warpAffine(marked_original, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # # Step 2: Remove noise and enhance
    # enhanced = remove_noise_and_enhance(deskewed)
    
    # # Step 3: Inpaint defects on color image
    # inpainted = inpaint_defects(img_rotated)
    
    # # Convert grayscale enhanced to color for comparison
    # enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # # Create side-by-side comparison
    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # # Original with marked defects
    # axes[0, 0].imshow(cv2.cvtColor(marked_original, cv2.COLOR_BGR2RGB))
    # axes[0, 0].set_title('Original Image (Defects Marked)', fontsize=14, fontweight='bold')
    # axes[0, 0].text(10, 30, 'Red boxes = Large defects', color='red', fontsize=10, 
    #                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # axes[0, 0].text(10, 60, 'Yellow circles = Noise spots', color='orange', fontsize=10,
    #                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # axes[0, 0].axis('off')
    
    # # Deskewed + Denoised
    # axes[0, 1].imshow(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB))
    # axes[0, 1].set_title('Cleaned & Enhanced (B&W)', fontsize=14, fontweight='bold')
    # axes[0, 1].axis('off')
    
    # # Inpainted color
    # axes[1, 0].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    # axes[1, 0].set_title('Defects Removed (Color)', fontsize=14, fontweight='bold')
    # axes[1, 0].axis('off')
    
    # # Defect mask visualization
    # defect_vis = cv2.applyColorMap(defect_mask, cv2.COLORMAP_JET)
    # axes[1, 1].imshow(cv2.cvtColor(defect_vis, cv2.COLOR_BGR2RGB))
    # axes[1, 1].set_title('Defect Detection Heatmap', fontsize=14, fontweight='bold')
    # axes[1, 1].axis('off')
    
    # plt.tight_layout()
    # plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # print(f"Comparison saved to: {output_path}")
    # plt.show()
    
    # # Save individual processed images
    # cv2.imwrite("marked_original.png", marked_original)
    # cv2.imwrite("enhanced_bw.png", enhanced)
    # cv2.imwrite("inpainted_color.png", inpainted)
    # cv2.imwrite("defect_mask.png", defect_mask)
    # print("Individual images saved: marked_original.png, enhanced_bw.png, inpainted_color.png, defect_mask.png")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "CO2022007_00028.TIF"
    # img.show(image_path)

    # --- 1. The Main Window Code (Simulated) ---
    print("Main application running...")
    img.show(image_path)

    # --- 2. Load the Image ---
    # Replace 'your_image.jpg' with the actual path to your image
    # # image_path = "your_image.jpg"
    # image_to_display = cv2.imread(image_path)

    # # # --- Fallback for testing if image file is not found ---
    # if image_to_display is None:
    #     print(f"Warning: Could not find '{image_path}'. Creating a dummy image instead.")
    #     # Create a simple 200x200 blue square image for demonstration
    #     image_to_display = np.zeros((200, 200, 3), dtype=np.uint8)
    #     image_to_display[:, :] = (255, 0, 0)  # BGR format (Blue)

    # # --- 3. Call the function to display the image in a separate window ---
    # print("Displaying image in a new window. Press any key in the image window to close it and continue the script.")
    # # display_image_in_new_window(image_to_display, window_name="Result - Deskewed Output")
    # img.show(image_to_display, window_name="Result - Deskewed Output")
    # # # --- 4. Continue Main Window Execution ---
    # # print("Image window closed. Main application continues execution.")


    # process_and_compare(image_path, output_path="before_after_comparison.png")