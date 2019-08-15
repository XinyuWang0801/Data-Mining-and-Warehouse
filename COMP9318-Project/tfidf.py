# coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer

# 语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print('word: \n',word)
# 查看词频结果
print('array: \n',X.toarray())

from sklearn.feature_extraction.text import TfidfTransformer

# 类调用
transformer = TfidfTransformer()
print('transformer: \n', transformer)
# 将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
# 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
weight = tfidf.toarray() # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
print('weight: \n', weight)
print('length of weight',len(weight))
for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
    for j in range(len(word)):
        print(word[j], weight[i][j])