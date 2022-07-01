import transformers
import torch.nn as nn
import config

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init()
        self.bert=transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )
        self.bert_drop=nn.Dropout(0.3)
        self.out=nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2= self.bert(
            ids,
            attention_masks=mask,
            token_type_ids=token_type_ids
        )