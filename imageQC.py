import filter as fltr
import image as img
#########################################################################################
if __name__ == "__main__":
    image_path ="Rotated_Page_25_Deg.tif" # "CO2022007_00028.TIF"
    img_gray=img.to_gray(image_path)
    img_no_noise=img.remove_noise(img_gray)
    # img_final, skew_angle = img.deskew(img_no_noise)
    img_final, skew_angle = img.deskew_using_text_bbox(img_no_noise)

    print(f"Detected skew angle: {skew_angle:.2f} degrees")
    img.show(img_final)
#########################################################################################
