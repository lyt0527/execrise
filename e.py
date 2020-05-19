# import pandas as pd	
# import altair as alt	
	
# data = pd.DataFrame({'country_id': [1, 2, 3, 4, 5, 6],	
#                      'population': [1, 100, 200, 300, 400, 500],	
#                      'income':     [1000, 50, 200, 300, 200, 150]})

# categorical_chart = alt.Chart(data).mark_circle(size=200).encode(	
#                         x='population:Q',	
#                         color='country_id:Q')import numpy as np
# import pymysql

# db = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="123456", db="crawler")

# cur = db.cursor()

# sql = "select comment_id, comment_content, author_realname, author_tel, order_number from comments limit 10"

# cur.execute(sql)
# results = cur.fetchall()
# print(results)

# db.close()
# for i in results:
# 	print(i) 
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import time
# opt = webdriver.ChromeOptions()
# url = "https://www.baidu.com/"
# opt.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"')
# opt.set_headless()

# driver = webdriver.Chrome(options=opt)
# driver.get(url)

# inputs = driver.find_element_by_id("kw")
# inputs.send_keys("python")
# inputs.send_keys(Keys.ENTER)
# time.sleep(1)
# driver.save_screenshot('1.png')
# driver.quit()
# import numpy as np
# a = np.random.randn(256)
# print(a.shape)