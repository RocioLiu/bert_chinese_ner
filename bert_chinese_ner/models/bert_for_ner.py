import torch
import torch as nn
import torch.nn.functional as F

from bert_chinese_ner.models.transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
# from .transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

from bert_chinese_ner.models.layers.crf import CRF

from bert_chinese_ner.models.transformers.models.bert.configuration_bert import BertConfig
config = BertConfig.from_pretrained("bert-base-uncased", author="DogeCheng")
print(config)

model = BertModel(config)
print(model)
for name, param in model.named_parameters():
    print(name, '\t', param.size())


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, num_tags):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                input_lens=None
                ):
        # (pooler) shape: (768,)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  #?




