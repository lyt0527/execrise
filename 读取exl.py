import xlwt
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("C:\\Users\\liuyuntao\\Desktop\\历史评论数据.xlsx")
file.to_csv("C:\\Users\\liuyuntao\\Desktop\\2.csv", index=False)
print("另存完成")

# file = pd.read_csv("C:\\Users\\liuyuntao\\Desktop\\2.csv")
# print(file.系统评分)
#删除表中含有任何NAN的行

# print(file.shape)
# print(file.ID, file.系统评分)
# X = file.ID
# y = file.系统评分
# print(y)
# plt.scatter(X, y)
# plt.show()
