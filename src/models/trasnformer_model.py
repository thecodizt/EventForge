import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers, max_event_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Embedding(max_event_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.max_event_length = max_event_length

    def forward(self, x):
        embedded = self.embedding(x)
        positions = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        pos_embedded = self.pos_encoder(positions)
        embedded = embedded + pos_embedded
        transformer_out = self.transformer_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        output = self.fc(transformer_out[:, -1, :])
        return output.view(-1, self.max_event_length, output.size(-1))

def create_model(config, vocab_size):
    return TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        max_event_length=config['data']['max_event_length']
    )