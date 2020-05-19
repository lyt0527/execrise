import pymysql

# db = pymysql.connect(host="172.22.14.51", port=8097, user="root", password="system", db="intelligent_marketing_platform")
db = pymysql.connect(host="172.22.14.51", port=8097, user="root", password="system", db="CGP")
cur = db.cursor()
# data表所有tdcid
# sql = "SELECT tdc_id FROM data;"
# cur.execute(sql)
# results = cur.fetchall()
# l = []
# for j in results:
# 	l.append(j[0])
# l = list(set(l))

# tdc_id
# sql1 = "SELECT user_id FROM tdc_id"
# cur.execute(sql1)
# res = cur.fetchall()
# l1 = []
# for k in res:
# 	l1.append(k[0])
# l1 = list(set(l1))

# 被点赞
# sql1 = "SELECT beidianzan, user_id FROM beidianzan;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_by_thumb_up_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")
	
# 被关注
# sql1 = "SELECT guanzhu, user_id FROM beiguanzhu;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_by_attention_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")
	
# 被评论
# sql1 = "SELECT beipinglun, user_id FROM beipinglun;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_by_comments_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")

# 被浏览
# sql1 = "SELECT liulanshu, user_id FROM by_liulan;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_by_browsing_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")

# 发帖
# sql1 = "SELECT fatieshu, user_id FROM fatie;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_posting_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")

# 关注
# sql1 = "SELECT guanzhu, user_id FROM guanzhu;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_follow_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")

# 评论
# sql1 = "SELECT pinglun, user_id FROM pinglun;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))
# for i in l1:
# 	sql3 = "update data set Forum_comment_number = (%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql3)
# 	# print(sql3)
# 	db.commit()
# 	print("更新完成")

# sql1 = "SELECT mobile, user_id FROM user;"
# cur.execute(sql1)
# re = cur.fetchall()
# l1 = []
# for k in re:
# 	l1.append(k)
# l1 = list(set(l1))

# for i in l1:
# 	sql2  = "update data set tel=(%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
# 	cur.execute(sql2)
# 	db.commit()
# 	print("更新完成")
	# else: 
	# 	sql3 = "update data set tel=(%s) where tdc_id=(%s);" % ('"'+str(i[0])+'"', '"'+str(i[1])+'"')
	# 	cur.execute(sql3)
	# 	# print(sql3)
	# 	db.commit()
	# 	print("更新完成1")
 	
# for i in l1:
# 	if i in l:
# 		print("已存在")
# 	else:
# 		sql2 = "insert into data(tdc_id) values (%s);" % ('"'+str(i)+'"')		
# 		cur.execute(sql2)
# 		db.commit()
# 		print("插入完成")

# 更新时间
# sql = "select * FROM time"
# cur.execute(sql)
# re = cur.fetchall()
# print(re)
# l1 = []
# for i in re:
# 	l1.append(i)
# for j in l1:
# 	# print(type(j[1]))
# 	sql2 = "update data set Store_app_activate_time=(%s) where tdc_id=(%s)" % ('"'+str(j[1])+'"', '"'+str(j[0])+'"')
# 	cur.execute(sql2)
# 	db.commit()
# 	print("更新完成")
db.close()

