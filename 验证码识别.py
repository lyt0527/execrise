from PIL import Image
import pytesseract

im = Image.open(r"C:\Users\liuyuntao\Desktop\exercise\验证码1.jpg")
# print(im)
#进行灰度值转换
im.convert("L")
# im1.save("1.png")
ret = pytesseract.image_to_string(im, lang='chi_sim', config="--psm 7")
print(ret)