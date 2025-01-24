import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import time
import json
import sys
from pathlib import Path

def report_progress(progress):
    """Report progress back to the API"""
    print(json.dumps({
        "status": "progress",
        "progress": progress,
        "state": "training"
    }))
    sys.stdout.flush()

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(MusicLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.lstm3 = nn.LSTM(hidden_size//2, hidden_size//2, batch_first=True)
        self.dropout3 = nn.Dropout(0.3)
        
        self.dense = nn.Linear(hidden_size//2, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, device=torch.device):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=0.0001
    )
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Report progress percentage
        progress = int((epoch / num_epochs) * 100)
        report_progress(progress)
        
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_val_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = total_val_loss / batch_count
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(json.dumps({
                "status": "early_stopping",
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }))
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def prepare_data(batch_size=32):
    # Specify the path to processed data in output directory
    data_path = Path('output/processed_music_data.pkl')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {data_path}")
    
    # Load preprocessed data
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading processed data: {str(e)}")
    
    # Prepare sequences
    X = data[:, :-1, :]
    y = data[:, 1:, :]
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Create dataset and loaders
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, X.shape[2]

def main():
    try:
        # Parse command line arguments
        params = json.loads(sys.argv[1])
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(json.dumps({
            "status": "info",
            "message": f"Using device: {device}"
        }))
        
        # Prepare data with batch size from parameters
        batch_size = params.get('batch_size', 32)
        train_loader, val_loader, input_size = prepare_data(batch_size)
        
        # Create model
        model = MusicLSTM(input_size=input_size)
        model = model.to(device)
        
        # Train model with epochs from parameters
        num_epochs = params.get('epochs', 100)
        model, train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader,
            num_epochs=num_epochs,
            device=device
        )
        
        # Save model with unique ID
        model_path = f"models/music_generator_{params['id']}.pth"
        Path("models").mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'params': params
        }, model_path)
        
        # Plot and save training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        history_path = f"public/training_history_{params['id']}.png"
        plt.savefig(history_path)
        plt.close()
        
        # Report completion
        print(json.dumps({
            "status": "complete",
            "model_path": model_path,
            "history_path": history_path,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1]
        }))
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))

if __name__ == "__main__":
    main()



# python3 src/python/train_model.py '{"id": "test1", "batch_size": 32, "epochs": 100}'
