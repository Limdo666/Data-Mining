#coding:utf-8
import jieba.posseg as pseg
from tqdm import tqdm

stop_words_file = 'stop_words_ch.txt'
#处理大文档
def readFile(addr):
    texts=""
    with open(addr,'rt',encoding="utf-8")as f:
         texts=f.readlines()
    #print(texts)
    return texts



def cutWord(lines):
    # 分词并保存长度大于等于2的词
    stop_words = loadStopWords(stop_words_file)
    fo = open("cutWord.txt",'r+',encoding="utf-8")
    for line in tqdm(lines):
        I = []
        nounsList = []
        words = pseg.cut(line)
        for word, flag in words:
            if len(word)>1 and word not in stop_words:
                I.append((flag, word))
        # 抽取名词
        for element in I:
            if element[0] == 'n':
                #nounsList.append(element[1])
                fo.write(str(element[1])+" ")
        fo.write("\n")
    fo.close()
    # 返回一个名词列表
    #return nounsList


def loadStopWords(filename):
    stop_words = set()
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words





if __name__ == "__main__":
    text = readFile("./news/原数据/文化.txt")
    cutWord(text)
    # print(cutWord(text))