import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, max_event_length):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, vocab_size)
        self.max_event_length = max_event_length

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        conv_results = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled_results = [nn.functional.max_pool1d(res, res.size(2)).squeeze(2) for res in conv_results]
        cat = torch.cat(pooled_results, 1)
        output = self.fc(cat)
        return output.view(-1, self.max_event_length, output.size(-1))

def create_model(config, vocab_size):
    return CNNModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        num_filters=config['model']['num_filters'],
        filter_sizes=config['model']['filter_sizes'],
        max_event_length=config['data']['max_event_length']
    )