try:
 	x = int(input("请输入一个数字: "))
except ValueError as e:
 	print("您输入的不是数字，请再次尝试输入！", e)
finally:
 	print("end")