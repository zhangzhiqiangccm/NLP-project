from trainer import TransformersTrainer
from model import BertForPTuning
import json
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MyTrainer(TransformersTrainer):
    def preprocess(self):
        self.label_dic = {'100': '民生', '101': '文化', '102': '娱乐', '103': '体育', '104': '财经', '106': '房产', '107': '汽车',
                          '108': '教育', '109': '科技', '110': '军事', '112': '旅游', '113': '国际', '114': '证券', '115': '农业',
                          '116': '游戏'}

        self.label_index = {k: i for i, (k, v) in enumerate(self.label_dic.items())}
        self.label_idx_dic = {k: self.tokenizer.convert_tokens_to_ids(list(v)) for k, v in self.label_dic.items()}

    def inputs_process(self, data_text, data_label):

        prompt = '播报一则体育新闻：'
        self.mask_pos = [5, 6]

        encodings = []
        labels = []
        for text, label in zip(data_text, data_label):
            text = prompt + text
            encoding = self.tokenizer_(text)
            encodings.append(encoding)
            labels.append(self.label_index[label])

        item = {}
        for encoding in encodings:
            for key in ['input_ids', 'attention_mask']:
                if key in item.keys():
                    item[key].append(encoding[key])
                else:
                    item[key] = [encoding[key]]

        for key in ['input_ids', 'attention_mask']:
            item[key] = torch.cat(item[key])

        return item, labels

    def get_train_data(self):
        text = []
        label = []
        with open('data/tnews_public/train.json', encoding='utf-8')as file:
            for line in file.readlines()[0:512]:
                line = line.strip()
                dic = json.loads(line)
                text.append(dic['sentence'])
                label.append(dic['label'])
        encoding, label = self.inputs_process(text, label)
        return DataLoader(Dataset(encoding, label), self.batch_size)

    def get_dev_data(self):
        text = []
        label = []
        with open('data/tnews_public/dev.json', encoding='utf-8')as file:
            for line in file.readlines()[0:100]:
                line = line.strip()
                dic = json.loads(line)
                text.append(dic['sentence'])
                label.append(dic['label'])
        encoding, label = self.inputs_process(text, label)
        return DataLoader(Dataset(encoding, label), self.batch_size)

    def configure_optimizer(self):
        self.prompt_optimizer = AdamW([{'params': self.model.prompt_embedding.parameters()},
                                       {'params': self.model.lstm_head.parameters()},
                                       {'params': self.model.mlp_head.parameters()}],
                                      lr=2e-4)
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data, mode):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        label_idx = self.label_idx_dic.values()
        label_idx_0, label_idx_1 = zip(*label_idx)

        token_logits = out[0]
        mask_logits = token_logits[:, self.mask_pos[0]][:, label_idx_0] + \
                      token_logits[:, self.mask_pos[1]][:, label_idx_1]
        y_pred = torch.argmax(mask_logits, dim=1)
        y_true = data['labels'].to(self.device)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(mask_logits, y_true)

        return loss, y_pred.cpu().numpy(), y_true.cpu().numpy()

    def train_func(self, loader):
        train_loss = 0
        all_label = []
        all_pred = []
        pbar = tqdm(loader)
        for batch in pbar:
            self.prompt_optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss, output, label = self.train_step(batch, mode='train')
            if output is not None and label is not None:
                all_label.extend(label)
                all_pred.extend(output)

            loss.backward()

            self.prompt_optimizer.step()
            self.optimizer.step()

            train_loss += loss.item()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        self.print_metrics('train', train_loss / len(loader), **self.calculate_metrics(all_label, all_pred))


# path = 'bert'
path = 'roberta_large'
model = BertForPTuning.from_pretrained(path, prompt_index=[1, 2, 3, 4])
trainer = MyTrainer(model, batch_size=8, lr=2e-5, max_length=32, model_path=path, do_dev=True, monitor='acc')
trainer.run()
