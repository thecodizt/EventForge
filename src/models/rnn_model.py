import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, max_event_length):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_event_length = max_event_length

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])
        return output.view(-1, self.max_event_length, output.size(-1))

def create_model(config, vocab_size):
    return RNNModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        max_event_length=config['data']['max_event_length']
    )
