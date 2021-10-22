import re
import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''


#连接图数据库
graph = Graph("http://localhost:7474",auth=("neo4j","demo"))


attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}

# 有的实体后面有括号，里面的内容可以作为标签
# 提取到标签后，把括号部分删除
def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)  # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)
    return x

#读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")  #取出三元组
        head = get_label_then_clean(head, label_data)
        relation_data[head][relation] = tail


#读取实体-属性-属性值三元组文件
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split("\t")  # 取出三元组
        entity = get_label_then_clean(entity, label_data)
        attribute_data[entity][attribute] = value

#构建cypher语句
cypher = ""
in_graph_entity = set()
for i, entity in enumerate(attribute_data):
    #为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    #将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        text += "%s:\'%s\',"%(attribute, value)
    text = text[:-1] + "}"  #最后一个逗号替换为大括号
    if entity in label_data:
        label = label_data[entity]
        #带标签的实体构建语句
        cypher += "CREATE (%s:%s %s)" % (entity, label, text) + "\n"
    else:
        #不带标签的实体构建语句
        cypher += "CREATE (%s %s)" % (entity, text) + "\n"
    in_graph_entity.add(entity)



#构建关系语句
for i, head in enumerate(relation_data):

    # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
    if head not in in_graph_entity:
        cypher += "CREATE (%s {NAME:'%s'})" % (head, head) + "\n"
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():

        # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
        if tail not in in_graph_entity:
            cypher += "CREATE (%s {NAME:'%s'})" % (tail, tail) + "\n"
            in_graph_entity.add(tail)

        #关系语句
        cypher += "CREATE (%s)-[:%s]->(%s)" % (head, relation, tail) + "\n"

# print(cypher)

#执行建表脚本
graph.run(cypher)

#记录我们图谱里都有哪些实体，哪些属性，哪些关系，哪些标签
data = defaultdict(set)
for head in relation_data:
    data["entitys"].add(head)
    for relation, tail in relation_data[head].items():
        data["relations"].add(relation)
        data["entitys"].add(tail)

for enti, label in label_data.items():
    data["entitys"].add(enti)
    data["labels"].add(label)

for enti in attribute_data:
    for attr, value in attribute_data[enti].items():
        data["entitys"].add(enti)
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
