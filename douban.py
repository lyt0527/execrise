# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:20:36 2019

@author: liuyuntao
"""

import urllib.request
#import requests
import re
import ssl
from lxml import etree
ssl._create_default_https_context = ssl._create_unverified_context

data = urllib.request.urlopen("https://read.douban.com/provider/all").read()
data = data.encoding("utf-8")
#url = "https://www.csdn.net/"
#headers = {"User-Agent":"Mozilla/5.0"}

#res = requests.get(url, headers=headers).encoding("utf-8")
#html = res.text
#print(html)
#parseHtml = etree.HTML(html)
#
#t_list = parseHtml.xpath('//div[@class="title"]/h2/a/text()')
#print(t_list)

#
pattern = '<div class="name">(.*?)</div>'
#pattern = '<a href=(.*?)</a>'

data1 = re.compile(pattern).findall(data)
print(data1)

#with open("出版社.txt", "w") as f:
#    for i in range(0, len(data1)):
#        f.write(data1[i] + "\n")