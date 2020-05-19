from elasticsearch import Elasticsearch

es = Elasticsearch('96.4.1.34:9200')
# es = Elasticsearch('http://222.217.61.75:9002')

# res = es.search(index="data*", body={})
res = es.search()
def es_search(phone):
      query_json = {
      "bool": {
            "must": {
                  "term": {
                        "phone": phone
                        }
                  }
            }
      }
 source_arr = ["name","ipgeolocation",]
 res = es.search(index="burin*", body={"query": query_json, "_source": source_arr})
 last_res = res['hits']['hits'][-1]
 # print('ladt_res',last_res)
 jwd_1 = last_res['_source']['ipgeolocation']
 # print(jwd_1)
 return jwd_1
# es.index(index="data1",doc_type="doc_type_test",body = action)
# print("上传完毕")

# 增
# body = {"name" : "lucy", "age" : "22", "sex" : "woman"}
# es.index(index="data1", doc_type="doc_type_test", id="111111", body={})
# print("增加完成")

# 删
# es.delete(index="data1", doc_type="doc_type_test", id="111111")
# print("删除完成")

# 改
# es.index(index="data1", doc_type="doc_type_test", id="111111", body={"id":"222"})
# print("修改完成")

# 查
# 打印出字段
for re in res:
      print(re)

data = (res["hits"])["hits"]
print(data)
# data = ((res["hits"])["hits"])
# for i in data:
#       id = (i["_source"])
#       print(id)
	# content = (i["_source"])["tags"]
	# print(id, content["content"], content["dominant_color_name"])
# print("查找完成")