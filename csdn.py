# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:34:42 2019

@author: liuyuntao
"""

#import requests
#import re
#
#url = "https://www.csdn.net/"
#headers = {"User-Agent" : "Mozilla/5.0"}
#
#res = requests.get(url, headers=headers).encoding("utf-8")
#html = res.text
#
##s = parseHtml.xpath('//div[@class="title"]/h2/a')
#pattern = '<a href=(.*?)</a>'
#
#data = re.compile(pattern).findall(res)
#print(data)
from bs4 import BeautifulSoup
from lxml import html
import xml
import requests

url = "https://www.csdn.net/"
data = requests.get(url)                 #Get该网页从而获取该html内容
#print(data.text)
soup = BeautifulSoup(data.text, "lxml")  #用lxml解析器解析该网页的内容, 好像f.text也是返回的html
#print(data.content.decode())								#尝试打印出网页内容,看是否获取成功
content = soup.find_all('div',class_="p12" )   #尝试获取节点，因为calss和关键字冲突，所以改名class_
print(content)

#for k in soup.find_all('div',class_='title'):#,找到div并且class为pl2的标签
#   a = k.find_all('h2')       #在每个对应div标签下找span标签，会发现，一个a里面有四组span
#   print(a[0].string)            #取第一组的span中的字符串
