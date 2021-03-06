import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from .layers.crf import CRF


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, pretrained_model_name, config, num_tags, batch_first=False):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(pretrained_model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=batch_first)
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
                a batch of input_ids with shape (max_seq_len, batch_size)
                if batch_first is `False` else (batch_size, max_seq_len)
            attention_mask: torch.Tensor[torch.Tensor[int]]
            token_type_ids: torch.Tensor[torch.Tensor[int]]
            labels: torch.Tensor[torch.Tensor[int]].
            input_lens:
        return:
        """
        # input: input_ids.shape: (max_seq_len, batch_size) if batch_first is False
        # bert_outputs: class BaseModelOutputWithPoolingAndCrossAttentions
        # o1: last_hidden_state (ie. sequence output)
        # o2: pooler_output
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output: (*input_ids, 768) = (max_seq_len, batch_size, 768)
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)
        # logits: (max_seq_len, batch_size, num_tags)
        logits = self.classifier(sequence_output)
        # outputs = (logits, )
        if label_ids is not None:
            # logits: (max_seq_length, batch_size, num_tags)
            loss = -1 * self.crf(emissions=logits, tags=label_ids, mask=attention_mask)
            outputs = logits, loss  # emissions, loss
        else:
            outputs = logits

        return outputs


