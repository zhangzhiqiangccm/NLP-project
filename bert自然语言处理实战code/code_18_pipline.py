# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:22:55 2020

@author: ljh
"""

from transformers import *

nlp = pipeline("sentiment-analysis")
print(nlp("I like this book!"))

##########################################feature-extraction
import numpy as np
nlp_features = pipeline('feature-extraction')
output = nlp_features('Code Doctor Studio is a Chinese company based in BeiJing.')
print(np.array(output).shape)   # (Samples, Tokens, Vector Size)(1, 16, 768)


############################掩码语言建模
nlp_fill = pipeline("fill-mask")
print(nlp_fill.tokenizer.mask_token)
print(nlp_fill(f"Li Jinhong wrote many {nlp_fill.tokenizer.mask_token} about artificial intelligence technology and helped many people."))




############################抽取式问答


nlp_qa = pipeline("question-answering")
print(nlp_qa(context='Code Doctor Studio is a Chinese company based in BeiJing.',
       question='Where is Code Doctor Studio?') )




###################################摘要

TEXT_TO_SUMMARIZE = '''
In this notebook we will be using the transformer model, first introduced in this paper. Specifically, we will be using the BERT (Bidirectional Encoder Representations from Transformers) model from this paper.
Transformer models are considerably larger than anything else covered in these tutorials. As such we are going to use the transformers library to get pre-trained transformers and use them as our embedding layers. We will freeze (not train) the transformer and only train the remainder of the model which learns from the representations produced by the transformer. In this case we will be using a multi-layer bi-directional GRU, however any model can learn from these representations.
'''
summarizer = pipeline('summarization')
print(summarizer(TEXT_TO_SUMMARIZE))


# #################命名实体识别

nlp_token_class = pipeline("ner")
print(nlp_token_class(
        'Code Doctor Studio is a Chinese company based in BeiJing.'))

