import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.lstm_model import create_model
from src.data.datasets.event_dataset import EventDataset

def train_model(config):
    # Load data
    dataset = EventDataset(config['data']['path'], config['data']['sequence_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Create model
    model = create_model(config, dataset.get_vocab_size())
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        for batch_input, batch_target in dataloader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Loss: {total_loss/len(dataloader):.4f}")

    # Save the model
    torch.save(model.state_dict(), 'results/models/lstm_chess_single_agent_1k_model.pth')