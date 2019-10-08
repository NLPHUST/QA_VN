import re
from collections import Counter
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import gensim
import numpy as np
def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s
def clean_text(text):
    text = text.lower()
    # text = remove_accents(text)
    text = word_tokenize(text)
    return text

def read_data_from_file(path_name):
    questions = []
    answers = []
    labels = []
    with open(path_name,'r',encoding='utf-8') as lines:
        for line in lines:
            tmp = line.strip().split('\t')
            questions.append(clean_text(tmp[0]))
            answers.append(clean_text(tmp[1]))
            labels.append(int(tmp[2]))
    return questions,answers,labels

def create_embedd(path,vocab,embed_size=300,is_binary=True,mode="other"):
    if(mode == "gensim"):
        model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(path,binary=is_binary)
    else:
        model_word2vec = {}
        with open(path,'r',encoding="utf-8") as lines:
            for line in lines:
                tmp = line.strip().split()
                vector = [float(w) for w in tmp[1:]]
                model_word2vec[tmp[0]] = vector
    embedding_matrix = np.zeros((len(vocab),embed_size))
    not_found_c = 0
    for i in range(len(vocab)):
        try:
            embedding_vector = model_word2vec[vocab[i]]
            embedding_matrix[i] = embedding_vector
        except:
            not_found_c = not_found_c + 1
            print("Not found {} word in embed".format(not_found_c))
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embed_size).astype("float32")
    return embedding_matrix
def creat_voc(data,min_count = 5):
    voc = set()
    all_words = []
    for sent in data:
        for w in sent:
            voc.add(w)
            all_words.append(w)
    counter = Counter(all_words)
    voc = list(voc)
    voc = [w for w in voc if counter[w] > min_count]
    voc.insert(0,'<PAD>')
    voc2index = {}
    for i in range(len(voc)):
        voc2index[voc[i]] = i
    return voc,voc2index
def convert_and_pad(X,voc2index,max_len):
    X_idx = [convert_data_to_index(x,voc2index) for x in X]
    X_pad = sequence.pad_sequences(X_idx,maxlen=max_len,padding='post',truncating="post")
    return X_pad

def convert_data_to_index(string_data, vocab):
    index_data = []
    for i in range(len(string_data)):
        if string_data[i] in vocab:
            index_data.append(vocab[string_data[i]])
    return index_data

def map_score(s1,s2,y_pred,labels):
    QA_pairs = {}
    for i in range(len(s1)):
        pred = y_pred[i]

        s1_str = " ".join(s1[i])
        s2_str = " ".join(s2[i])
        if s1_str in QA_pairs:
            QA_pairs[s1_str].append((s2_str, labels[i], pred[-1]))
        else:
            QA_pairs[s1_str] = [(s2_str, labels[i], pred[-1])]

    MAP, MRR = 0, 0
    num_q = len(QA_pairs.keys())
    for s1_str in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[s1_str] = sorted(QA_pairs[s1_str], key=lambda x: x[-1], reverse=True)

        for idx, (s2_str, label, prob) in enumerate(QA_pairs[s1_str]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)
        if(p==0):
            AP = 0
            num_q = num_q - 1
        else:
            AP /= p
        MAP += AP
    MAP /= num_q
    MRR /= num_q
    return MAP,MRR
