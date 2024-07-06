import torch
from torch.utils.data import Dataset
import pandas as pd
from constants import ROOT_DIR

class EventDataset(Dataset):
    def __init__(self, csv_file, sequence_length):
        self.data = pd.read_csv(f'{ROOT_DIR}/data/processed' + csv_file)
        self.sequence_length = sequence_length
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)
        self.index_to_token = {v: k for k, v in self.vocab.items()}

    def build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for _, row in self.data.iterrows():
            vocab.setdefault(row['event_type'], len(vocab))
            vocab.setdefault(row['agent_id'], len(vocab))
            vocab.setdefault(row['context'], len(vocab))
        return vocab

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx+self.sequence_length]
        target = self.data.iloc[idx+self.sequence_length]

        input_seq = []
        for _, event in sequence.iterrows():
            event_encoding = self.encode_event(event)
            input_seq.append(event_encoding)

        input_tensor = [torch.tensor(seq).clone().detach() for seq in input_seq]
        target_tensor = self.encode_event(target)

        return input_tensor, target_tensor, sequence, target

    def encode_event(self, event):
        event_encoding = [
            self.vocab.get(event['event_type'], self.vocab['<UNK>']),
            self.vocab.get(event['agent_id'], self.vocab['<UNK>']),
            self.vocab.get(event['context'], self.vocab['<UNK>'])
        ]
        return torch.tensor(event_encoding)

    def decode_event(self, encoded_event):
        decoded = {}
        decoded['event_type'] = self.index_to_token.get(encoded_event[0].item(), '<PAD>')
        decoded['agent_id'] = self.index_to_token.get(encoded_event[1].item(), '<PAD>')
        decoded['context'] = self.index_to_token.get(encoded_event[2].item(), '<PAD>')
        return decoded