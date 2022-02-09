import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_chinese_ner.models.transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
# from .transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

from bert_chinese_ner.models.layers.crf import CRF
from .layers.crf import CRF

from bert_chinese_ner.models.transformers.models.bert.configuration_bert import BertConfig
from .transformers.models.bert.configuration_bert import BertConfig

from bert_chinese_ner import ner_config
from .. import ner_config



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
                label_ids=None
                ):
        """
        Args:
            input_ids: torch.Tensor[torch.Tensor[int]].
                a batch of input_ids with shape (batch_size, seq_len).
            attention_mask: torch.Tensor[torch.Tensor[int]]
            token_type_ids: torch.Tensor[torch.Tensor[int]]
            labels: torch.Tensor[torch.Tensor[int]].
            input_lens:
        return:
        """
        # (pooler) shape: (768,)
        # outputs: class BaseModelOutputWithPoolingAndCrossAttentions
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output: [*input_ids, 768] = [batch_size, max_seq_len, 768]
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits, )
        if label_ids is not None:
            # logits: (seq_length, batch_size, num_tags)
            loss = self.crf(emissions = logits, tags=label_ids, mask=attention_mask)
            # loss, scores
            outputs = (-1*loss, ) + outputs

        return outputs



input_ids = aa['input_ids']
attention_mask = aa['attention_mask']
token_type_ids = aa['token_type_ids']
label_ids = aa['label_ids']

input_ids.shape # (batch_size, max_len) = (64, 128)
num_tags = len(ner_config.LABELS)



config = BertConfig.from_pretrained(ner_config.BASE_MODEL_NAME)
print(config)

bert = BertModel(config)
dropout = nn.Dropout(config.hidden_dropout_prob)
classifier = nn.Linear(config.hidden_size, num_tags)
crf = CRF(num_tags, batch_first=True)


# --
bert_outputs = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
type(bert_outputs)
bert_outputs[0].shape

sequence_output = bert_outputs[0]
sequence_output = dropout(sequence_output)
print(sequence_output.shape) # (64, 128, 768)

logits = classifier(sequence_output)
print(logits.shape) # (64, 128, 7)

outputs = (logits, )
loss = crf(emissions = logits, tags=label_ids, mask=attention_mask)
outputs = (-1*loss, ) + outputs
