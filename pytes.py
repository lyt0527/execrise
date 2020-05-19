# import pytesseract
# from PIL import Image
# # pytesseract.pytesseract.tesseract_cmd='D:\Program Files\python\Tesseract-OCR\\tesseract.exe'
# def getyzm():
#     image1 = Image.open('C:\\Users\\liuyuntao\\Desktop\\2\\2.jpg')
#     w,h = image1.size
#     #创建新图片
#     image2 = Image.new("RGB",(w+10,h+6),(255,255,255))
#     #两张图片相加： 我这里的图片不是标准的图片格式所以需要盖在新图片上
#     image2.paste(image1,(5,3))
#     # image2.save("yzm.png")
#     result = pytesseract.image_to_string(image2,lang="num")
#     return result
 
# print(getyzm())
# import datetime as dt

# now_time = dt.datetime.now()
# print(now_time.hour, now_time.minute, now_time.second)

# str_time = dt.datetime.strftime(now_time, "%Y-%m-%d %H:%M:%S")[-8:]

# print(str_time)
# import tensorflow as tf

# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[5, 6], [7, 8]])

# c = tf.matmul(a, b)
# with tf.Session() as sess:
# 	print(sess.run(c))
import requests
from lxml import etree
from selenium import webdriver

url = "https://club.autohome.com.cn/bbs/forum-c-4890-1.html"
opt = webdriver.ChromeOptions()
opt.set_headless()

driver = webdriver.Chrome(options=opt)
driver.get(url)

title = driver.find_element_by_xpath('//ul[@class="post-list"]//p[@class="post-title"]/a').text
print(title)