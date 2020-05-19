import pymysql
import pandas as pd
import math

# conn = pymysql.connect(host="127.0.0.1", user='root', password="123456", db="111")
conn = pymysql.connect(host="172.22.14.51", port=8097, user="root", password="system", db="caijingbo")

sql = "select * from data_7_23"
data = pd.read_sql(sql, conn)
data.to_csv("C:\\Users\\liuyuntao\\Desktop\\data.csv")
print("保存完成")
# pd.set_option('display.max_rows', None)
# print(data.id)
