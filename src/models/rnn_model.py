import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim * 3, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size, sequence_length, event_length = x.shape
        embedded = self.embedding(x)
        embedded = embedded.view(batch_size, sequence_length, -1)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = rnn_out[:, -1, :]
        output = self.fc(rnn_out)
        return output

def create_model(config, vocab_size):
    return RNNModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    )
