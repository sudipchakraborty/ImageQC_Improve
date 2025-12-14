import filter as fltr
import image as img
#########################################################################################
if __name__ == "__main__":
    image_path = "CO2022007_00028.TIF"
    img_gray=img.to_gray(image_path)
    img_final=img.remove_noise(img_gray)
    img.show(img_final)
#########################################################################################
