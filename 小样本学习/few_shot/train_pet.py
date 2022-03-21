import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.models.bert import BertForMaskedLM
from utils import fix_seed
from trainer import TransformersTrainer
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, target_ids, data_length):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.target_ids = target_ids
        self.data_length = data_length

    def __getitem__(self, idx):
        if self.target_ids is None:
            return self.input_ids[idx], self.attention_masks[idx]
        else:
            return self.input_ids[idx], self.attention_masks[idx], self.target_ids[idx]

    def __len__(self):
        return self.data_length


class MyTrainer(TransformersTrainer):

    def preprocess(self):
        self.prefix = '很好，'
        self.mask_index = 2
        self.pos_id = self.tokenizer.convert_tokens_to_ids('好')
        self.neg_id = self.tokenizer.convert_tokens_to_ids('差')

    def pattern_data(self, data, label):
        all_input_ids = []
        all_attention_mask = []
        all_target_ids = []
        for d in zip(data, label):
            # [CLS] ..... [SEP]
            text = self.prefix + d[0]
            encoding = self.tokenizer_(text, return_tensors='np')
            # 输入值的下标
            input_ids = encoding['input_ids'][0]
            # 输出值的下标
            target_ids = [-100] * len(input_ids)
            # 添加随机mask
            # input_ids, target_ids = random_mask(input_ids, self.tokenizer)
            attention_mask = encoding['attention_mask'][0]

            # positive
            if d[1] == 1:
                input_ids[self.mask_index] = self.tokenizer.mask_token_id
                target_ids[self.mask_index] = self.pos_id
            # negative
            elif d[1] == 0:
                input_ids[self.mask_index] = self.tokenizer.mask_token_id
                target_ids[self.mask_index] = self.neg_id

            all_input_ids.append(torch.tensor(input_ids))
            all_attention_mask.append(torch.tensor(attention_mask))
            all_target_ids.append(torch.tensor(target_ids))

        return all_input_ids, all_attention_mask, all_target_ids

    def get_train_data(self):
        train_text = []
        train_label = []
        with open('data/sentiment/sentiment.train.data', encoding='utf-8')as file:
            for line in file.readlines()[0:20]:
                t, l = line.strip().split('\t')
                train_text.append(t)
                train_label.append(int(l))

        train_dataset = Dataset(*self.pattern_data(train_text, train_label), len(train_text))
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def get_dev_data(self):
        dev_text = []
        dev_label = []
        with open('data/sentiment/sentiment.valid.data', encoding='utf-8')as file:
            for line in file.readlines():
                t, l = line.strip().split('\t')
                dev_text.append(t)
                dev_label.append(int(l))

        dev_dataset = Dataset(*self.pattern_data(dev_text, dev_label), len(dev_text))
        return DataLoader(dev_dataset, batch_size=self.batch_size)

    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data, mode):
        input_ids = data[0].to(self.device).long()
        attention_mask = data[1].to(self.device).long()
        target_ids = data[2].to(self.device).long()

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=target_ids)
        if mode == 'dev':
            # [batch_size, seq_len, vocab_size]
            output = outputs.logits

            labels = target_ids[:, self.mask_index].cpu().numpy()

            label_dict = {self.pos_id: 1, self.neg_id: 0}
            labels = [label_dict[l] for l in labels]

            pos_logits = output[:, self.mask_index, self.pos_id].unsqueeze(0)
            neg_logits = output[:, self.mask_index, self.neg_id].unsqueeze(0)

            # 这里需要注意位置，neg才是0
            logits = torch.cat([neg_logits, pos_logits], dim=0)
            y_pred = torch.argmax(logits, dim=0).cpu().numpy()

            return outputs.loss, y_pred, labels
        return outputs.loss

    def train_func(self, loader):
        pbar = tqdm(loader)
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self.train_step(batch, mode='train')
            loss.backward()
            self.optimizer.step()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')


def main():
    fix_seed(2021)

    max_length = 64
    batch_size = 32
    lr = 1e-5

    model_path = 'E:\\ptm\\roberta'
    model = BertForMaskedLM.from_pretrained(model_path)
    trainer = MyTrainer(model, batch_size=batch_size, lr=lr, max_length=max_length, model_path=model_path,
                        do_train=True, do_dev=True, do_test=False, test_with_label=False)
    trainer.configure_metrics(do_acc=True, do_f1=False, do_recall=False, do_precision=False)
    trainer.run()


if __name__ == '__main__':
    main()
