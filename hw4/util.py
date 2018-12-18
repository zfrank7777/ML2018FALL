import jieba
import pickle
import time
import math
import pandas as pd
import numpy as np


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


def w2index(text_set):
    maxlength = 128
    with open('word_set_no_seg.pickle', 'rb') as handle:
        word_set = pickle.load(handle)

    vec = []
    for text in text_set:
        # sentence = parse_sentence(text)
        sentence = text
        x = []
        for word in sentence:
            if word in word_set:
                x.append(word_set.index(word))
            else:
                x.append(7744)  # UNK
        x.append(7745)  # EOS
        # padding
        while len(x) < maxlength:
            x.append(7746)
        if len(x) > maxlength:
            x = x[:maxlength-1]
            x.append(7745)
        x = np.array(x)
        vec.append(x)
    vec = np.array(vec)
    return vec


if __name__ == "__main__":
    pass
