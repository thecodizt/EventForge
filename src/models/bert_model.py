import torch
import torch.nn as nn
from transformers import BertModel

class BertModel(nn.Module):
    def __init__(self, pretrained_bert_name, num_classes):
        super(BertModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(last_hidden_state)
        return logits

def create_model(config, num_classes):
    return BertModel(
        pretrained_bert_name=config['model']['pretrained_bert_name'],
        num_classes=num_classes
    )
