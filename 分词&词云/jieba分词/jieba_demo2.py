#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba

#加载自定义词典
jieba.load_userdict("user_dict.txt")

#采用默认方式也就是我们的精确模式
words = jieba.cut("大鹏主演屌丝男士上了微博热搜")
print("/".join(words))