# -*- coding: utf-8 -*-
# __author__ = 'lidong'

"""
Description:
"""

from os.path import isdir, isfile, join
from os import listdir
import jieba.posseg as pseg
from collections import Counter, defaultdict
from numba import jit
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import MySQLdb
from tqdm import tqdm
from bidict import bidict

label_no = bidict({'财经':1,
            '军事':2,
            '社会':3,
            '生活':4,
            '文化':5,
            '汽车':6,
            '娱乐':7,
            '体育':8,
            '科技':9,
            '健康':10
            })


flags = ('n', 'nt', 'nl', 'nz','ng')

stop_words_file = 'stop_words_ch.txt'


def loadOriginalDataTxt(base_path):
    """
    从文件夹中读取文本数据，每类文件放在同一个文件夹中
    :param base_path:
    :return:
    """
    folders = [f for f in listdir(base_path) if isdir(join(base_path, f))]
    for folder in folders:
        # 打开一个文件夹，每个文件是一个样本
        files = [f for f in listdir(join(base_path, folder)) if f.endswith('.txt') and isfile(join(base_path, folder, f))]
        for file in files:
            # 读取一个样本
            with open(join(base_path, folder, file), 'r', encoding='utf8') as f:
                txt = ''.join(f.readlines())
                label = label_no.get(folder, None)
                if label:
                    # 返回标签和内容
                    yield label, txt

def cutWord(base_path, bags, stop_words):
    stop_words = loadStopWords(stop_words_file)
    for label, txt in loadOriginalDataTxt(base_path):
        words = set(extract_word(txt, flags, stop_words)).intersection(list(bags))
        yield label, list(words)

def loadDataFromTxt(category):
    if not isfile(category + '_cut.txt'):
        return []
    with open(category + '_cut.txt') as file:
        for line in file:
            if len(line) < 10:
                continue
            yield category, line.strip().split()

def loadStopWords(filename):
    stop_words = set()
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words

def loadOriginalDataSQL(labels):
    """
    将数据从数据库取出，并使用jieba分词，然后将原文章和分词存入文件
    :param labels:
    :return:
    """
    conn = MySQLdb.connect(host="localhost",db="mydb",user="root",password="donglsky",charset = 'utf8',use_unicode = True)
    cur = conn.cursor()
    select_sql = """SELECT content from news where category = %s"""
    # 停用词
    stop_words = loadStopWords(stop_words_file)
    for label in labels:
        print('read data: {} from SQL...'.format(label))
        try:
            cur.execute(select_sql, [label])
            contents = cur.fetchall()
            original_doc = open(label+'.txt','wt',encoding='utf-8')
            cut_words = open(label+'_cut.txt','wt',encoding='utf-8')
            for content in tqdm(contents):
                content = content[0]
                try:
                    words = pseg.cut(content)
                    original_doc.write(content+'\n')
                    words = [word.word for word in words if word.flag in flags and word.word not in stop_words]
                    words = [word for word in words if len(word) > 1]
                    if len(words) >= 5:
                        cut_words.write(' '.join(words)+'\n')
                except Exception as e:
                    print('Error:', e)
                    continue
            original_doc.close()
            cut_words.close()
        except Exception as e:
            print("Label: {} can't be queried.")
            print(e)

def extract_word(s, flags=flags, stop_words=set()):
    words = pseg.cut(s)
    return (word.word for word in words if word.flag in flags and word.word not in stop_words)


if __name__ == '__main__':
    # words, labels, frequency = extractWords('newstrain', flags=flags, save=True, train=True)
    # bag = constructBag(labels, frequency)
    pass