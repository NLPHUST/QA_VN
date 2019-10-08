import json
import re
import os
import gensim
from gensim.models import Word2Vec,KeyedVectors
import time
import pandas as pd
from nltk.tokenize import word_tokenize
from utils import *
texts= []
def process(text):
    text = text.lower()
    text = word_tokenize(text)
    return text
root_dir = "./data/general"
list_file = os.listdir(root_dir)
for f in list_file:
    with open(os.path.join(root_dir,f),"r",encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line.strip())
            try:
                text = row["question"]
                text = process(text)
                texts.append(text)
                text = row["answer"]
                text = process(text)
                texts.append(text)
            except:
                continue
print('Load data finshed')
print("num_text : ",len(texts))
s_time = time.time()
model = Word2Vec(texts,min_count=2,size=300)
model.wv.save_word2vec_format('word_vector.bin',binary=True)
print("Finished !")
print("Time train /s : ",time.time()-s_time)
model_word2vec = gensim.models.KeyedVectors.load_word2vec_format("word_vector.bin",binary=True)
vocab = model_word2vec.vocab.keys()
print(len(vocab))
w ="hỏng"
if(w in vocab):
   print(model_word2vec.most_similar(positive=w))
