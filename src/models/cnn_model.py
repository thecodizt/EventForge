import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, max_event_length):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(max_event_length * embedding_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, vocab_size)
        self.max_event_length = max_event_length

    def forward(self, x):
        batch_size, sequence_length, event_length = x.shape
        embedded = self.embedding(x)  # (batch_size, sequence_length, event_length, embedding_dim)
        embedded = embedded.view(batch_size, sequence_length, -1)  # (batch_size, sequence_length, event_length * embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, event_length * embedding_dim, sequence_length)
        conv_results = [F.relu(conv(embedded)) for conv in self.convs]
        pooled_results = [F.max_pool1d(res, res.size(2)).squeeze(2) for res in conv_results]
        cat = torch.cat(pooled_results, 1)
        output = self.fc(cat)
        return output

def create_model(config, vocab_size):
    return CNNModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        num_filters=config['model']['num_filters'],
        filter_sizes=config['model']['filter_sizes'],
        max_event_length=config['data']['max_event_length']
    )