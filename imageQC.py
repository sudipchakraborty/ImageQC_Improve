# Requires: pip install opencv-python scikit-image numpy matplotlib
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt

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
    """Process image and create side-by-side comparison"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Deskew
    deskewed, angle = deskew_image(gray)
    print(f"Detected skew angle: {angle:.2f} degrees")
    
    # Rotate original color image by same angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Step 2: Remove noise and enhance
    enhanced = remove_noise_and_enhance(deskewed)
    
    # Step 3: Inpaint defects on color image
    inpainted = inpaint_defects(img_rotated)
    
    # Convert grayscale enhanced to color for comparison
    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Deskewed + Denoised
    axes[0, 1].imshow(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Deskewed + Noise Removed + Enhanced', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Inpainted color
    axes[1, 0].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Defects Inpainted (Color)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Zoomed comparison (center crop)
    h, w = img.shape[:2]
    crop_size = min(h, w) // 3
    y_start, x_start = h//3, w//3
    orig_crop = img[y_start:y_start+crop_size, x_start:x_start+crop_size]
    enh_crop = enhanced[y_start:y_start+crop_size, x_start:x_start+crop_size]
    
    # Create zoomed comparison
    zoom_compare = np.hstack([
        cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.cvtColor(enh_crop, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    ])
    axes[1, 1].imshow(zoom_compare)
    axes[1, 1].set_title('Zoomed: Original (Left) vs Enhanced (Right)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_path}")
    plt.show()
    
    # Save individual processed images
    cv2.imwrite("enhanced_bw.png", enhanced)
    cv2.imwrite("inpainted_color.png", inpainted)
    print("Individual images saved: enhanced_bw.png, inpainted_color.png")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "CO2022007_00028.TIF"
    process_and_compare(image_path, output_path="before_after_comparison.png")