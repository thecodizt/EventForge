import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Embedding(1000, embedding_dim)  # Arbitrary max sequence length for positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim * 3, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embedding_dim * 3, vocab_size)

    def forward(self, x):
        batch_size, sequence_length, event_length = x.shape
        embedded = self.embedding(x)
        positions = torch.arange(0, sequence_length, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embedded = self.pos_encoder(positions).unsqueeze(2)
        embedded = embedded + pos_embedded
        embedded = embedded.view(batch_size, sequence_length, -1)
        transformer_out = self.transformer_encoder(embedded.permute(1, 0, 2))
        transformer_out = transformer_out[-1]
        output = self.fc(transformer_out)
        return output

def create_model(config, vocab_size):
    return TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers']
    )
