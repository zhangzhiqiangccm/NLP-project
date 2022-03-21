from transformers.models.bert import BertTokenizerFast
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report


class TransformersTrainer:

    def __init__(self, model, batch_size, lr, max_length, model_path, epochs=100,
                 do_train=True, do_dev=True, do_test=False,
                 test_with_label=False, save_model_name='best_model.p', attack=False,
                 monitor='loss'):
        """
        :param model: 模型
        :param batch_size: 批次大小
        :param lr: 学习率
        :param max_length: 序列的最大长度
        :param model_path: 预训练模型的存储路径
        :param epochs: 训练轮数
        :param do_train: 是否训练
        :param do_dev: 是否验证
        :param do_test: 是否测试
        :param test_with_label: 测试时是否有标签
        :param save_model_name: 保存的模型名
        :param attack: 是否做对抗训练
        :param monitor: 保存模型的监控指标
        """
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.train()
        self.optimizer = self.configure_optimizer()
        self.do_train = do_train
        self.do_dev = do_dev
        self.do_test = do_test
        self.test_with_label = test_with_label
        self.epochs = epochs
        self.save_model_name = save_model_name
        self.attack = attack
        self.monitor = monitor

        self.preprocess()

        if do_train:
            self.train_loader = self.get_train_data()

        if do_dev:
            self.dev_loader = self.get_dev_data()

        if do_test:
            self.test_loader = self.get_test_data()

        self.configure_metrics()

    def preprocess(self):
        pass

    def get_train_data(self):
        raise NotImplementedError

    def get_dev_data(self):
        raise NotImplementedError

    def get_test_data(self):
        pass

    def configure_optimizer(self):
        raise NotImplementedError

    def train_step(self, data, mode):
        raise NotImplementedError

    def predict_step(self, data):
        return []

    def adversarial(self, data):
        print('adversarial not implemented')

    def tokenizer_(self, text, text_pair=None, return_tensors='pt', truncation=True, padding='max_length'):
        return self.tokenizer(text=text,
                              text_pair=text_pair,
                              return_tensors=return_tensors,
                              truncation=truncation,
                              padding=padding,
                              max_length=self.max_length)

    def train_func(self, loader):
        train_loss = 0
        all_label = []
        all_pred = []
        pbar = tqdm(loader)
        for batch in pbar:
            self.optimizer.zero_grad()
            loss, output, label = self.train_step(batch, mode='train')
            if output is not None and label is not None:
                all_label.extend(label)
                all_pred.extend(output)

            loss.backward()

            if self.attack:
                self.adversarial(batch)

            self.optimizer.step()

            train_loss += loss.item()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        self.print_metrics('train', train_loss / len(loader), **self.calculate_metrics(all_label, all_pred))

    def dev_func(self, loader, mode):
        dev_loss = 0

        all_label = []
        all_pred = []
        metrics = {}
        for batch in tqdm(loader):
            with torch.no_grad():
                loss, output, label = self.train_step(batch, mode)
                dev_loss += loss.item()
                if output is not None:
                    all_label.extend(label)
                    all_pred.extend(output)
        # 打印评价指标
        if all_pred is not None:
            metrics = self.calculate_metrics(all_label, all_pred)
            self.print_metrics(mode, dev_loss / len(loader), **metrics)
            if self.print_report:
                target_names = [f'class {i}' for i in range(len(set(all_label)))]
                print(classification_report(all_label, all_pred, target_names=target_names))

        # 返回monitor指标
        if self.monitor == 'loss':
            return dev_loss / len(loader)
        elif output is not None:
            return metrics[self.monitor]

    def predict_func(self, loader):
        all_out = []
        for batch in tqdm(loader):
            with torch.no_grad():
                output = self.predict_step(batch)
                all_out.extend(output.tolist())
        return all_out

    def print_metrics(self, mode, loss, acc, recall, precision, f1):
        result_str = f'{mode} loss:{loss:.4f}'
        if self.do_acc:
            result_str += f' acc:{acc:.4f}'
        if self.do_recall:
            result_str += f' recall:{recall:.4f}'
        if self.do_precision:
            result_str += f' precision:{precision:.4f}'
        if self.do_f1:
            result_str += f' f1:{f1:.4f}'

        print(result_str)

    def configure_metrics(self, do_acc=True, do_recall=False, do_precision=False, do_f1=False, print_report=False):
        self.do_acc = do_acc
        self.do_recall = do_recall
        self.do_precision = do_precision
        self.do_f1 = do_f1
        self.print_report = print_report

    def calculate_metrics(self, y_true, y_pred):
        acc = 0
        recall = 0
        precision = 0
        f1 = 0
        if self.do_acc:
            acc = accuracy_score(y_true, y_pred)
        if self.do_recall:
            recall = recall_score(y_true, y_pred, average='macro')
        if self.do_precision:
            precision = precision_score(y_true, y_pred, average='macro')
        if self.do_f1:
            f1 = f1_score(y_true, y_pred, average='macro')
        return {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1}

    def run(self):
        if self.monitor == 'loss':
            dev_metric = float('inf')
        else:
            dev_metric = 0
        for epoch in range(self.epochs):
            print(f'***********epoch: {epoch + 1}***********')
            if self.do_train:
                self.train_func(self.train_loader)
                if not self.do_dev:
                    torch.save(self.model, self.save_model_name)
                    print('save model')
            if self.do_dev:
                metric = self.dev_func(self.dev_loader, mode='dev')
                if self.monitor == 'loss':
                    if dev_metric > metric:
                        dev_metric = metric
                        torch.save(self.model, self.save_model_name)
                        print('save model')
                else:
                    if dev_metric < metric:
                        dev_metric = metric
                        torch.save(self.model, self.save_model_name)
                        print('save model')
                print(f'best {self.monitor}:{dev_metric}')
            if self.do_test:
                self.model.eval()
                if self.test_with_label:
                    self.dev_func(self.test_loader, mode='test')
                else:
                    y_pred = self.predict_func(self.test_loader)
                    return y_pred
