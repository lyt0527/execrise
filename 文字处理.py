from snownlp import SnowNLP, seg, sentiment
import jieba

# data = SnowNLP("质量不太好")

# print(data.words)

# print(list(jieba.cut("质量不太好")))
# print(data.pinyin)
# print(list(data.tags))

text = '''
自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，
而在于研制能有效地实现自然语言通信的计算机系统，
特别是其中的软件系统。因而它是计算机科学的一部分。
'''

s = SnowNLP(text)

print(s.words)
print(list(jieba.cut(text)))
print(s.sentiments)
#提取关键句子
print(s.summary())

ss = "今天天气不错"
sss = SnowNLP(ss)
print(sss.words)
print(list(jieba.cut(ss)))
print(sss.sentiments)

seg_list = jieba.cut('我来到北京清华大学', cut_all=True)
print('Full Mode:', '/'.join(seg_list))  # 全模式
seg_list = jieba.cut('我来到北京清华大学', cut_all=False)
print('Default Mode:', '/'.join(seg_list))  # 精确模式
seg_list = jieba.cut('他来到了网易杭研大厦')  # 默认是精确模式
print('/'.join(seg_list))
seg_list = jieba.cut_for_search('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')  # 搜索引擎模式
print('搜索引擎模式:', '/'.join(seg_list))
seg_list = jieba.cut('小明硕士毕业于中国科学院计算所，后在日本京都大学深造', cut_all=True)
print('Full Mode:', '/'.join(seg_list))