import jieba
from jieba import posseg

# 不使用jieba可以换成数组
# def segment_line(line):
#     tokens = jieba.cut(line, cut_all=False)
#     return " ".join(tokens)
#
#
# def process_line(line):
#     if isinstance(line, str):
#         tokens = line.split("|")
#         result = [segment_line(t) for t in tokens]
#         return " | ".join(result)


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence: 带分割的句子
    :param cut_type: 'word' use jieba.lcut; 进行分词 'char' use list(sentence) 按每一个单词分开
    :param pos: enable POS  是否返回词性
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)
