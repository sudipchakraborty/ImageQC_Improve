# Requires: pip install opencv-python scikit-image numpy pytesseract
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
# import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def deskew_image(img_gray):
    edges = cv2.Canny(img_gray,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    if lines is None:
        return img_gray
    angles = []
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        angles.append(angle)
    median_angle = np.median(angles)
    # rotate by negative median_angle
    (h,w) = img_gray.shape
    M = cv2.getRotationMatrix2D((w//2,h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def despeckle_and_preserve_punct(img_gray):
    # bilateral or non-local means
    den = cv2.fastNlMeansDenoising(img_gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # binarize using Sauvola
    window_size = 25
    thresh_s = threshold_sauvola(den, window_size=window_size)
    bw = (den > thresh_s).astype(np.uint8)*255
    # morphological opening to remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # preserve tiny components that look like punctuation: check small CCs from original bw
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    result = opened.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 10 and h <= 6 and w <= 6:
            # likely punctuation — re-add from bw
            result[labels==i] = 255
    return result

def inpaint_black_holes(img_color, mask_thresh=10):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mask = (gray < mask_thresh).astype(np.uint8)*255
    # dilate mask a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    # inpaint using Telea
    inpainted = cv2.inpaint(img_color, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

# Example pipeline
if __name__ == "__main__":
    img = cv2.imread("CO2022007_00028.TIF")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    deskewed = deskew_image(gray)
    despeckled = despeckle_and_preserve_punct(deskewed)
    # combine despeckled with original color for further inpainting
    color_rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2,img.shape[0]//2), np.median([0]), 1.0), (img.shape[1],img.shape[0]))
    inpainted = inpaint_black_holes(color_rot)
    cv2.imwrite("deskewed_bw.png", despeckled)
    cv2.imwrite("inpainted_color.png", inpainted)
    print("OCR sample:", pytesseract.image_to_string(deskewed))
