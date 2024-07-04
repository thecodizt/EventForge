import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, max_event_length):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.max_event_length = max_event_length

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # [CLS] token representation
        logits = self.fc(pooled_output)
        return logits
    
def create_model(config, vocab_size):
    return BERTModel(
        vocab_size=vocab_size,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        max_event_length=config['data']['max_event_length']
    )

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')