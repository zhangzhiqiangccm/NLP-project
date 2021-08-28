# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 06:15:51 2020

@author: ljh
"""



from transformers import (GPT2LMHeadModel, GPT2Config,CONFIG_NAME, AdamW, BertTokenizer)
import json
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
from itertools import chain
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DialogDataset(Dataset):

    def __init__(self, tokenizer, max_history=15, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(DialogDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.target_file = kwargs['data_path']
        SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
        self.bos, self.eos, self.speaker1, self.speaker2 = tokenizer.convert_tokens_to_ids(
                                        SPECIAL_TOKENS)

    def _get_line(self, index):
        with open(self.target_file, "r", encoding="utf-8") as f:
            f.seek(index)
            line = f.readline()
        return line
    
    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dialog = tokenize(dialog)
        history = dialog[:-1]
        candidates = dialog[-1]
        return self.process(history, candidates)

    def process(self, history, resposne, with_eos=True):
        
        sequence = [[self.bos]] + history + [resposne + ([self.eos] if with_eos else [])]
        sequence = [sequence[0]] + [[self.speaker2 if i % 2 else self.speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.bos] + [self.speaker2 if i % 2 else self.speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels
    
    
print("Prepare datasets") 
model_checkpoint = 'cgpt'
tokenizer = BertTokenizer(os.path.join(model_checkpoint, "vocab.txt"), do_lower_case=True)

valid_path="dataGPT/toy_valid.txt"
valid_dataset = DialogDataset(tokenizer, data_path=valid_path) 
input_idssz = tokenizer.convert_ids_to_tokens(valid_dataset[0]['input_ids'])
token_type_idssz = tokenizer.convert_ids_to_tokens(valid_dataset[0]['token_type_ids'])
lm_labelssz = tokenizer.convert_ids_to_tokens(valid_dataset[0]['lm_labels'])
print('输入文本:',''.join(input_idssz))
print('段编码:',''.join(token_type_idssz))
print('标签:',''.join(lm_labelssz))

train_path="dataGPT/toy_train.txt"
train_dataset = DialogDataset(tokenizer, data_path=train_path)

config = GPT2Config.from_json_file(os.path.join(model_checkpoint, 'config.json'))
model = GPT2LMHeadModel(config)
model.to(device)  
    





train_batch_size = 2
valid_batch_size = 2
num_workers = 0

  




train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              num_workers=num_workers,
                              batch_size=train_batch_size,
                              shuffle=True
                              )
valid_loader = DataLoader(valid_dataset,
                          collate_fn=valid_dataset.collate,
                          num_workers=num_workers,
                          batch_size=valid_batch_size,
                          shuffle=False)


lr = 5e-5
optimizer = AdamW([{'params': model.parameters(), 'initial_lr': lr}], lr=lr, correct_bias=True)

lossfun = torch.nn.CrossEntropyLoss(ignore_index=-1)

def evaltest():
    model.eval()
    allloss = []
    for batch in valid_loader:
        with torch.no_grad():
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(device
                                                         ) for input_tensor in batch)
            lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].reshape(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            loss = lossfun(lm_logits_flat_shifted, lm_labels_flat_shifted)
            allloss.append(lossfun)
    return np.mean(allloss)


gradient_accumulation_steps = 16

model_size = 768
warmup_steps = 5000
from_step = -1
max_epochs =80
noam_lambda = lambda step: (
        model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)))
noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=from_step)

logdir='./modelGPT/'
for epoch in range(max_epochs):
    model.train()
    allloss = []
    for iteration,batch in enumerate( train_loader):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(device) for input_tensor in batch)
        lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        lm_logits_flat_shifted = lm_logits[..., :-1, :].reshape(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        lm_loss = lossfun(lm_logits_flat_shifted, lm_labels_flat_shifted)
        
        loss = lm_loss / gradient_accumulation_steps
        loss.backward()
        allloss.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            noam_scheduler.step()
            optimizer.zero_grad()
            
    evalloss = evaltest()
    print('train:',np.mean(allloss), "lr",optimizer.param_groups[0]['lr'],'eval',evalloss)
    torch.save({'state_dict': model.cpu().state_dict()}, os.path.join(logdir, 'model_training.bin'))
    model.config.to_json_file(os.path.join(logdir, CONFIG_NAME))
    tokenizer.save_vocabulary(logdir)
