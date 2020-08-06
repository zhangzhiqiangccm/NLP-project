#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba

#使用精确模式
words = jieba.cut("我毕业于北京理工大学")
print("采用精确模式来进行分词")
print("/".join(words))

words = jieba.cut("我毕业于北京理工大学",cut_all=True)
print("采用全模式来进行分词")
print("/".join(words))

#使用搜索引擎模式
print("搜索引擎模式")
words = jieba.cut_for_search("我毕业于北京理工大学，后就职于中国科学院计算技术研究所。")
print("/".join(words))
