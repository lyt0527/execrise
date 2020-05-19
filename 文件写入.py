# file = "1111"
# with open("./1.txt", "a+") as f:
# 	f.write(file)
	# f.write("\n")
# f = open("./1.txt", "a+")
# f.write("111")
# 读取图片
# import numpy as np
# from PIL import Image

# photo = Image.open("C:\\Users\\liuyuntao\\Desktop\\1.jpg")
# data = np.array(photo)
# print(data.shape)
# import requests

# url = 'http://www.iqiyi.com/v_19ruldtq84.html'
# res = requests.get(url)
# print(res.status_code)
# 'http://autocomment.mop.com/comment/api/tt/mopauto/190903114618578/commentreply?softtype=mopauto&softname=mopauto_pc&rowkey=190903114618578&userid=mop_454402415&ts=1575341320&sign=91f3bf4ab9cb4b40db7091253ec2c362'
# import random,string
import requests, re
# import time
# headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36"}
# t = str(time.time()).split('.')[0]
# url = 'http://auto.mop.com/a/190903114618578.html'
# res = requests.get(url, headers=headers)
# print(res.url, res.headers)
# a = re.split('\/|\.', url)[-2]
# num=string.ascii_lowercase+string.digits
# b = "".join(random.sample(num,32))
# url = 'http://autocomment.mop.com/comment/api/tt/mopauto/{}/commentreply?softtype=mopauto&softname=mopauto_pc&rowkey={}&userid=mop_454402415&ts={}&sign={}'.format(a, a, t, b)
# print(url)

# import requests


# headers = {
# 	'User-Agent':"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
# 	'Referer':'http://auto.mop.com/a/190905105018279.html'
# } 

# url = 'https://zhidao.baidu.com/api/qbpv?q=687131159163748972'

# res = requests.get(url, headers=headers)
# print(res.text)
# a = '<span class="question-all-answers-title" style="font-weight: 700;">40个回答</span>'
# b = re.findall(r'<span class="question-all-answers-title" .*>(\d+).*', a)[0]
# print(b)
# url = 'http://autocomment.mop.com/comment/api/tt/mopauto/190905105018279/commentreply?callback=jQuery18305373745940266936_1575442083943&softtype=mopauto&softname=mopauto_pc&to=http%3A%2F%2Fauto.mop.com%2Fa%2F190905105018279.html&aid=190905105018279&rowkey=190905105018279&userid=mop_454402415&username=Akakad&userpic=http%3A%2F%2Fi4.mopimg.cn%2Fhead%2F454402415%2F100x100&hotnum=10&commtype=0&depth=5&revnum=5&endkey=0&limitnum=10&ts=1575442084&sign=597c2470448670ea96a04b07d919cfaa&_=1575442084031'

# res = requests.get(url, headers=headers)
# print(res.text)
# a = '{"start_time": "2019-12-03 00:00:00","end_time": "2019-12-03 08:00:00","project_id":"app2.0","table_name":"user_zy_12043"}'
# b = '/root/bak/streamsets-datacollector-3.11.0/bin/streamsets cli -U http://96.4.0.10:18630 -u admin -p admin manager start -n userburiedtestzyc56426d7-3361-4194-9db2-12098797121d -R %s' % a
# print(b.split('\r\n'))
# print("x\ay")
a = '22222'
print(''.join(a))