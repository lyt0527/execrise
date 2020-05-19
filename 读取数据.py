# import pymysql
# import pandas as pd
# db = pymysql.Connect(host = '172.22.14.51', port = 8097, user = 'root', password = 'system', db = 'intelligent_marketing_platform')
# db = pymysql.Connect(host = '172.22.14.124', port = 8097, user = 'pachong', password = 'He7KzJXAKsBB8UyCFrpvCr4zWjFLDjTx', db = 'crawle')

# data = pd.read_sql('select * from comment', db)
# data = pd.read_csv('C:\\Users\\liuyuntao\\Desktop\\数据.csv')
# print(data)

# data.to_csv('C:\\Users\\liuyuntao\\Desktop\\数据.csv')
# print("保存完成")
# l = []
with open(r'C:\Users\liuyuntao\Desktop\1.txt', 'r') as f:
    a = f.readlines()
    for i in a:
        l.append(i.split('\n')[0].strip(''))
print(l)
