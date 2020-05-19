# import os
# import sys
# import paramiko
# from PyQt5 import QtCore, QtGui, QtWidgets

# if __name__ == '__main__':
# 	app = QtWidgets.QApplication(sys.argv)
# 	MyUI = QtWidgets.QWidget()
# 	MyUI.setWindowTitle("demo")
# 	MyUI.resize(400, 200)
# 	MyUI.show()
# 	sys.exit(app.exec_())
# Build a dataflow graph.
import re

# ret = re.search(r"\d+", "阅22读次数为 9999")
# print(ret.group())

ret = re.sub(r"\d+", '998', "python = 997 Java = 993")
print(ret)