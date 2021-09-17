import os
import zipfile
import urllib.request

from Word2Vec import *


# import pymongo
# db = pymongo.MongoClient().travel.articles
# class texts:
#     def __iter__(self):
#         for t in db.find().limit(30000):
#             yield t['words']

# Download a small chunk of Wikipedia articles collection
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

wv = Word2Vec(text_words, model='cbow', nb_negative=16,
              shared_softmax=True, epochs=2)  # 建立并训练模型
wv.save_model('myvec')  # 保存到当前目录下的myvec文件夹

# 训练完成后可以这样调用
wv = Word2Vec()  # 建立空模型
wv.load_model('myvec')  # 从当前目录下的myvec文件夹加载模型

wv.most_similar('five')
