from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BertForPTuning(BertPreTrainedModel):
    def __init__(self, config: BertConfig, prompt_index):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.prompt_index = prompt_index

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.prompt_embedding = torch.nn.Embedding(len(prompt_index), config.hidden_size)
        self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,
                                       hidden_size=config.hidden_size,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(config.hidden_size, config.hidden_size))
        self.init_weights()

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder.weight = new_embeddings.weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.set_output_embeddings(self.bert.embeddings.word_embeddings)

        # 替换embedding
        replace_embedding = self.prompt_embedding(torch.arange(len(self.prompt_index)).to(input_ids.device))[None, :]
        replace_embedding = self.lstm_head(replace_embedding)[0]
        replace_embedding = self.mlp_head(replace_embedding)
        raw_embedding = self.bert.embeddings.word_embeddings(input_ids)
        raw_embedding[:, self.prompt_index, :] = replace_embedding
        inputs_embeds = raw_embedding

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.cls(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            s = attention_mask.shape[0] * attention_mask.shape[1]
            loss = loss_fct(logits.view(s, -1), labels.view(-1))

        # token out / pool out / cls
        output = (logits,) + outputs[1:] + (outputs[0][:, 0],)
        return ((loss,) + output) if loss is not None else output
