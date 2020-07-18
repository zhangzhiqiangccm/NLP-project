
EMBEDDING_DIM = 100 # 如果用预训练的词向量的话，这里需要和预训练词向量维度一致
POS_SIZE = 82  # attention: 这里是用于pos_embedding,因为前面pos标记用到了 0 - 81共82个数据,num_embeddings = 82
POS_DIM = 25
HIDDEN_DIM = 200
BATCH = 128
EPOCHS = 100

config={}
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['BATCH'] = BATCH
config['PER_PRINT'] = 20
config["pretrained"]=False
config['EPOCHS'] = EPOCHS
config['LEARNING_RATE'] = 0.0005


