#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import FMM
import BMM

#使用双向最大匹配算法实现中文分词
words_dic = []

def init():
    """
    读取词典文件
    载入词典
    :return:
    """
    with open("dic.txt","r", encoding="utf8") as dic_input:
        for word in dic_input:
            words_dic.append(word.strip())

#实现双向匹配算法中的切词方法
def cut_words(raw_sentence,words_dic):
    bmm_word_list = BMM.cut_words(raw_sentence,words_dic)
    fmm_word_list = FMM.cut_words(raw_sentence,words_dic)
    bmm_word_list_size = len(bmm_word_list)
    fmm_word_list_size = len(fmm_word_list)
    if bmm_word_list_size != fmm_word_list_size:
        if bmm_word_list_size < fmm_word_list_size:
            return bmm_word_list
        else:
            return fmm_word_list
    else:
        FSingle = 0
        BSingle = 0
        isSame = True
        for i in range(len(fmm_word_list)):
            if fmm_word_list[i] not in bmm_word_list:
                isSame = False
            if len(fmm_word_list[i])  == 1:
                FSingle = FSingle + 1
            if len(bmm_word_list[i]) == 1:
                BSingle = BSingle + 1
        if isSame:
            return fmm_word_list
        elif BSingle > FSingle:
            return fmm_word_list
        else:
            return bmm_word_list


def main():
    """
    于用户交互接口
    :return:
    """
    init()
    while True:
        print("请输入您要分词的序列")
        input_str = input()
        if not input_str:
            break
        result = cut_words(input_str,words_dic)
        print("分词结果")
        print(result)

if __name__ == "__main__":
    main()
