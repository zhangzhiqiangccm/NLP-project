import pandas as pd

init_tags = pd.read_table('000.txt', header=None, sep=' ')

terms = []
current_tag = 'O'
current_term = ''
for row in init_tags.iterrows():
    word = row[1][0]
    tag = row[1][1]
    if tag == current_tag:
        current_term += word
    else:
        if current_tag != 'O':
            terms.append((current_term, current_tag))
        if tag != 'O':
            current_term = word
        else:
            current_term = ''
        current_tag = tag
print(terms)

sentences = []
current_sentence = ''
current_term_tag = 'Loc'
current_terms = []
current_tags = []
for term_tag in terms:
    term = term_tag[0]
    tag = term_tag[1]
    if tag == 'Loc' and tag != current_term_tag:
        sentences.append((current_sentence, current_terms, current_tags))
        print((current_sentence, current_terms, current_tags))
        current_sentence = term
        current_terms = []
        current_tags = []
        current_terms.append(term)
        current_tags.append(tag)
    else:
        current_sentence += term
        current_terms.append(term)
        current_tags.append(tag)
    current_term_tag = tag

res = []
for sentence in sentences:
    init_sentence = sentence[0]
    for cur_term in sentence[1][1:]:
        temp = 'L'
        for term ,tag in zip(sentence[1][1:], sentence[2][1:]):
            if term == cur_term:
                temp += tag[0]
            else:
                temp += term
        res.append((init_sentence, temp))

df = pd.DataFrame(res, columns=['初始句子', '生成句子'])
df.to_csv('res1.csv', index=False)
