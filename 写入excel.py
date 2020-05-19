import xlwt

wb = xlwt.Workbook(encoding="utf8")

sheet = wb.add_sheet("sheet", cell_overwrite_ok=True)

sheet.write(0, 0, '姓名')
sheet.write(0, 1, '年龄')

# data_row = 1
sheet.write(1, 0, '张三')
sheet.write(1, 1, '23')

wb.save("C:/Users/liuyuntao/Desktop/1.xls")