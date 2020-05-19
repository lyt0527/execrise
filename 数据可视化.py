#数据预处理流程
'''
	2019-06-10
	liuyuntao
'''
import matplotlib.pyplot as plt
import xlwt
import pandas as pd
import pymysql
import seaborn as sns 


# train = pd.read_excel("C:\\Users\\liuyuntao\\Desktop\\train.xls", rows=15000)
# train = pd.read_excel("C:\\Users\\liuyuntao\\Desktop\\train.xls", usecols=["ID", "订单号", "用户真实姓名", "用户电话号码"])
test = pd.read_excel("C:\\Users\\liuyuntao\\Desktop\\数据.xlsx")
test = test.dropna()
# t_test = test.发送时间
# list = []
# for t in t_test:
# 	t1 = float(t.split('-')[2])
	
# 	list.append(t1)
# print(list)

#拆出一部分当成测试数据
# train.to_excel("C:\\Users\\liuyuntao\\Desktop\\test.xls")
# print(train.head(10))
# print(train.shape)
# new_test = test.dropna()
# 查看列名 pandas.core.frame.DataFrame
# print(new_test)
# matplotlib
plt.scatter(test.发送时间, test.发送数量)
plt.plot(test.发送时间, test.发送数量)
plt.show()
# Seaborn
# sns.distplot(new_test.发送时间, new_test.发送数量)
# sns.plt.show()
# #查看个字段信息
# print(train.info())
# #查看数据的大体情况
# 
# 读取数据库
# conn = pymysql.connect(host="127.0.0.1", user='root', password="123456", db="crawler")
# sql = "select * from news"
# data = pd.read_sql(sql, conn)
# data.to_csv("C:\\Users\\liuyuntao\\Desktop\\sql.csv")
# data_ = pd.read_csv("C:\\Users\\liuyuntao\\Desktop\\sql.csv", usecols=["id", "comment_id", "comment_content"])
# print(data_.id)
# print(data_.head(10))
