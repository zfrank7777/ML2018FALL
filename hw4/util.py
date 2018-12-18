import jieba
from gensim.models import Word2Vec
import pickle
import time
import math
import pandas as pd
import numpy as np
model = Word2Vec.load("word2vec_model_128_0")


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def read_text(textfile):
    names = list("abc")
    df = pd.read_csv(textfile, delimiter=',',
                     encoding='utf-8', names=names)
    length = len(df['a'])
    text = []
    for i in range(1, length):
        if not isinstance(df['b'][i], str):
            text.append("")
        elif not isinstance(df['c'][i], str):
            text.append(df['b'][i])
        else:
            text.append(str(df['b'][i])+"，"+str(df['c'][i]))
    return text


def read_label(labelfile):
    df = pd.read_csv(labelfile, encoding='utf-8')
    labels = df['label'].values
    labels = np.array(labels)
    return labels


def parse_sentence(sentence):
    parsed = jieba.cut(sentence, cut_all=False)
    return parsed


def w2v(text, UNK, EOS):
    vecs = []
    lengths = []
    maxlength = 128
    for i in range(len(text)):
        # words = parse_sentence(text[i])
        words = text[i]
        sen_vec = []
        for each in words:
            if each not in model.wv.vocab:
                continue
            else:
                sen_vec.append(model[each])
        
        if len(sen_vec) > maxlength-1:
            sen_vec = sen_vec[:maxlength-1]
            sen_vec.append(EOS)
            lengths.append(maxlength)
        else:
            sen_vec.append(EOS)
            lengths.append(len(sen_vec))
            while len(sen_vec) < maxlength: 
                sen_vec.append(np.zeros(128))
        sen_vec = np.array(sen_vec)
        if sen_vec.shape[1] != maxlength:
            print ("shape not expected: ", sen_vec.shape)
        vecs.append(np.array(sen_vec))
    vecs = np.array(vecs)
    lengths = np.array(lengths)
    return vecs


if __name__ == "__main__":
    pass
