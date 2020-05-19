import pytesseract
from PIL import Image

image = Image.open("C:\\Users\\liuyuntao\\Desktop\\exercise\\验证码1.jpg")
string = pytesseract.image_to_string(image, lang='bm', config="--psm 7")
print(string)






