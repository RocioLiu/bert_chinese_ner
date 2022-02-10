import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_chinese_ner.models.transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
# from .transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

from bert_chinese_ner.models.layers.crf import CRF
from .layers.crf import CRF


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
                a batch of input_ids with shape (max_seq_len, batch_size).
            attention_mask: torch.Tensor[torch.Tensor[int]]
            token_type_ids: torch.Tensor[torch.Tensor[int]]
            labels: torch.Tensor[torch.Tensor[int]].
            input_lens:
        return:
        """
        # input: input_ids.shape: [max_seq_len, batch_size]
        # bert_outputs: class BaseModelOutputWithPoolingAndCrossAttentions
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output: [*input_ids, 768] = [max_seq_len, batch_size, 768]
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits, )
        if label_ids is not None:
            # logits: (max_seq_length, batch_size, num_tags)
            loss = self.crf(emissions = logits, tags=label_ids, mask=attention_mask)

        return -1*loss, outputs


## ---

# bert = BertModel(config)
# dropout = nn.Dropout(config.hidden_dropout_prob)
# classifier = nn.Linear(config.hidden_size, num_tags)
# crf = CRF(num_tags=num_tags, batch_first=True)
#
# # forward
# batch = next(iter(train_dataloader))
# batch.keys()
#
# input_ids = batch['input_ids'].transpose(0, 1)
# attention_mask = batch['attention_mask'].transpose(0, 1)
# token_type_ids = batch['token_type_ids'].transpose(0, 1)
# label_ids = batch['label_ids'].transpose(0, 1)
#
#
# bert_outputs = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
# sequence_output = bert_outputs[0]
# sequence_output.shape # torch.Size([128, 64, 768])
# sequence_output = dropout(sequence_output)
# logits = classifier(sequence_output)
# logits.shape # torch.Size([128, 64, 7])
# outputs = (logits, )
#
# attention_mask.shape
# loss = crf(emissions = logits, tags=label_ids, mask=attention_mask)



