import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import json

from constants import ROOT_DIR

class EventDataset(Dataset):
    def __init__(self, csv_file, sequence_length, vocab=None):
        self.data = pd.read_csv(f'{ROOT_DIR}/data/processed' + csv_file)
        self.sequence_length = sequence_length
        self.vocab = vocab if vocab else self.build_vocab()
        self.vocab_size = len(self.vocab)

    def build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for _, row in self.data.iterrows():
            vocab.setdefault(row['event_type'], len(vocab))
            context = json.loads(row['context'])
            for key, value in context.items():
                vocab.setdefault(key, len(vocab))
                vocab.setdefault(str(value), len(vocab))
        return vocab

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx+self.sequence_length]
        target = self.data.iloc[idx+self.sequence_length]

        input_seq = []
        for _, event in sequence.iterrows():
            event_encoding = [self.vocab.get(event['event_type'], self.vocab['<UNK>'])]
            context = json.loads(event['context'])
            for key, value in context.items():
                event_encoding.append(self.vocab.get(key, self.vocab['<UNK>']))
                event_encoding.append(self.vocab.get(str(value), self.vocab['<UNK>']))
            
            # Convert to one-hot encoding
            one_hot = F.one_hot(torch.tensor(event_encoding), num_classes=self.vocab_size).float()
            input_seq.append(one_hot.sum(dim=0))  # Sum to get a single vector per event

        # Pad sequence if necessary
        if len(input_seq) < self.sequence_length:
            padding_length = self.sequence_length - len(input_seq)
            input_seq.extend([torch.zeros(self.vocab_size)] * padding_length)

        input_tensor = torch.stack(input_seq)
        
        target_encoding = self.vocab.get(target['event_type'], self.vocab['<UNK>'])
        target_tensor = torch.tensor(target_encoding, dtype=torch.long)

        return input_tensor, target_tensor
    
    def get_vocab_size(self):
        return self.vocab_size
