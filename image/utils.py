import cv2
import numpy as np
import tkinter as tk

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
