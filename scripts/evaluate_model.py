import yaml
import torch
from torch.utils.data import DataLoader
from src.models.lstm_model import create_model
from src.data.datasets.event_dataset import EventDataset

def evaluate_model(config):
    # Load data
    dataset = EventDataset(config['data']['path'], config['data']['sequence_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Load model
    model = create_model(config, dataset.get_vocab_size())
    model.load_state_dict(torch.load('results/models/lstm_model.pth'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_input, batch_target in dataloader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            
            outputs = model(batch_input)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_target.size(0)
            correct += (predicted == batch_target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

if __name__ == "__main__":
    with open('configs/model_configs/lstm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    evaluate_model(config)