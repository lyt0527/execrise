from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# 生成随机字母
def rndChar():
	return chr(random.randint(65, 90))

# 随机颜色
def rndColor():
	return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

#随机颜色
def rndColor2():
	return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

width = 60 * 4
height = 60
image = Image.new("RGB", (width, height), (255, 255, 255))
#创建font对象,注意a是小写
font = ImageFont.truetype("arial.ttf", 36)
#创建Draw对象
draw = ImageDraw.Draw(image)
#填充四个像素
for x in range(width):
	for y in range(height):
		draw.point((x, y), fill=rndColor())

#输出文字
for i in range(4):
	draw.text((60 * i + 10, 10), rndChar(), font=font, fill=rndColor2())

#模糊
image = image.filter(ImageFilter.BLUR)
image.save("./验证码1.png", "jpeg")