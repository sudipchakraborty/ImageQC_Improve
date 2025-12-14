import filter as fltr
import image as img
#########################################################################################
if __name__ == "__main__":
    image_path = "CO2022007_00028.TIF"
    img_gray=img.to_gray(image_path)
    img_no_noise=img.remove_noise(img_gray)
    img_final, skew_angle = img.deskew(img_no_noise)
    print(f"Detected skew angle: {skew_angle:.2f} degrees")
    img.show(img_final)
#########################################################################################
