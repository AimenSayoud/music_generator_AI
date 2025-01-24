import torch
import numpy as np
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import pretty_midi
import zlib
from torch.utils.data import TensorDataset, DataLoader, Subset

class MusicLSTM(torch.nn.Module):
    # Keep your existing model class unchanged
    def __init__(self, input_size, hidden_size=384):
        super(MusicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm1 = torch.nn.LSTM(input_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        
        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = torch.nn.Dropout(0.2)
        
        self.lstm3 = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.dropout3 = torch.nn.Dropout(0.2)
        
        self.note_layer = torch.nn.Linear(hidden_size, input_size)
        self.velocity_layer = torch.nn.Linear(hidden_size, input_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        notes, velocities = x[:, :, :self.input_size], x[:, :, self.input_size:]
        x = torch.cat([notes, velocities], dim=-1)
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        note_out = self.sigmoid(self.note_layer(x))
        velocity_out = self.sigmoid(self.velocity_layer(x))
        
        return note_out, velocity_out

class MusicModelValidator:
    # Keep your existing validator class unchanged
    def __init__(self, model, device, fs=32):
        self.model = model
        self.device = device
        self.fs = fs
        self.metrics_history = {}
    
    def validate_step(self, batch_x, batch_y):
        self.model.eval()
        with torch.no_grad():
            notes_x = batch_x[:, :, :self.model.input_size].to(self.device)
            vel_x = batch_x[:, :, self.model.input_size:].to(self.device)
            
            pred_notes, pred_vels = self.model(batch_x.to(self.device))
            
            metrics = {
                'note_accuracy': self._calculate_note_accuracy(pred_notes, batch_y[:, :, :self.model.input_size]),
                'velocity_mse': self._calculate_velocity_mse(pred_vels, batch_y[:, :, self.model.input_size:]),
                'rhythm_metrics': self._calculate_rhythm_metrics(pred_notes),
                'density_metrics': self._calculate_density_metrics(pred_notes, batch_y[:, :, :self.model.input_size])
            }
            return metrics
    
    # Keep all helper methods unchanged
    def _calculate_note_accuracy(self, pred, target):
        """Calculate accuracy of note predictions"""
        pred_binary = (pred > 0.5).float()
        return accuracy_score(
            target.cpu().numpy().flatten(),
            pred_binary.cpu().numpy().flatten()
        )
    
    def _calculate_velocity_mse(self, pred, target):
        """Calculate MSE of velocity predictions"""
        return mean_squared_error(
            target.cpu().numpy(),
            pred.cpu().numpy()
        )
    
    def _calculate_rhythm_metrics(self, predictions):
        """Calculate rhythm-related metrics"""
        pred_np = predictions.cpu().numpy()
        
        # Beat alignment score
        beat_positions = np.arange(pred_np.shape[1]) % self.fs
        beats = beat_positions == 0
        beat_notes = np.mean(pred_np[:, beats, :])
        
        # Note density per beat
        densities = np.sum(pred_np > 0.5, axis=2)  # Sum across pitch dimension
        density_per_beat = np.mean(densities[:, beats])
        
        return {
            'beat_alignment': float(beat_notes),
            'density_per_beat': float(density_per_beat)
        }
    
    def _calculate_density_metrics(self, pred, target):
        """Calculate note density metrics"""
        pred_np = (pred > 0.5).cpu().numpy()
        target_np = target.cpu().numpy()
        
        pred_density = np.mean(np.sum(pred_np, axis=2))
        target_density = np.mean(np.sum(target_np, axis=2))
        
        return {
            'pred_density': float(pred_density),
            'target_density': float(target_density),
            'density_ratio': float(pred_density / max(target_density, 1e-5))
        }
    
    def validate_model(self, val_loader, save_dir=None):
        """Full model validation"""
        all_metrics = []
        
        for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
            metrics = self.validate_step(batch_x, batch_y)
            all_metrics.append(metrics)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} validation batches...")
        
        # Average metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        # Save and plot if directory provided
        if save_dir:
            self._save_validation_results(avg_metrics, save_dir)
        
        return avg_metrics
    
    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics across batches"""
        avg_metrics = {
            'note_accuracy': np.mean([m['note_accuracy'] for m in metrics_list]),
            'velocity_mse': np.mean([m['velocity_mse'] for m in metrics_list]),
            'rhythm': {
                'beat_alignment': np.mean([m['rhythm_metrics']['beat_alignment'] for m in metrics_list]),
                'density_per_beat': np.mean([m['rhythm_metrics']['density_per_beat'] for m in metrics_list])
            },
            'density': {
                'pred_density': np.mean([m['density_metrics']['pred_density'] for m in metrics_list]),
                'target_density': np.mean([m['density_metrics']['target_density'] for m in metrics_list]),
                'density_ratio': np.mean([m['density_metrics']['density_ratio'] for m in metrics_list])
            }
        }
        return avg_metrics
    
    def _save_validation_results(self, metrics, save_dir):
        """Save validation results and plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save metrics to JSON
        with open(save_dir / 'validation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualization
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Note Accuracy
        axs[0, 0].bar(['Note Accuracy'], [metrics['note_accuracy']])
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].set_title('Note Prediction Accuracy')
        
        # Plot 2: Velocity MSE
        axs[0, 1].bar(['Velocity MSE'], [metrics['velocity_mse']])
        axs[0, 1].set_title('Velocity Prediction MSE')
        
        # Plot 3: Rhythm Metrics
        rhythm_metrics = [
            metrics['rhythm']['beat_alignment'],
            metrics['rhythm']['density_per_beat']
        ]
        axs[1, 0].bar(['Beat Alignment', 'Density per Beat'], rhythm_metrics)
        axs[1, 0].set_title('Rhythm Metrics')
        
        # Plot 4: Density Comparison
        density_metrics = [
            metrics['density']['pred_density'],
            metrics['density']['target_density']
        ]
        axs[1, 1].bar(['Predicted Density', 'Target Density'], density_metrics)
        axs[1, 1].set_title('Note Density Comparison')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'validation_metrics.png')
        plt.close()

def load_processed_data():
    """Load and split data exactly like in training"""
    with open('processed_music_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Decompress sequences
    sequences = [
        np.frombuffer(zlib.decompress(seq), dtype=np.uint8).reshape(-1, 128)
        for seq in data['sequences']
    ]
    velocities = [
        np.frombuffer(zlib.decompress(vel), dtype=np.float32).reshape(-1, 128)
        for vel in data['velocities']
    ]

    # Create input/output pairs
    X = np.concatenate([np.array(sequences)[:, :-1], np.array(velocities)[:, :-1]], axis=-1)
    y = np.concatenate([np.array(sequences)[:, 1:], np.array(velocities)[:, 1:]], axis=-1)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Create dataset
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Recreate validation split (same as training)
    indices = list(range(len(full_dataset)))
    np.random.seed(42)  # Match training random seed
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(full_dataset)))
    val_idx = indices[:split]  # Use first 20% for validation

    # Create validation subset
    val_data = Subset(full_dataset, val_idx)
    return val_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        checkpoint = torch.load('music_generator_model.pth', map_location=device)
        model = MusicLSTM(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Create validation loader (aligned with training split)
        val_data = load_processed_data()
        val_loader = DataLoader(
            val_data,
            batch_size=32,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Create validator and run validation
        validator = MusicModelValidator(model, device)
        metrics = validator.validate_model(
            val_loader,
            save_dir='validation_results'
        )
        
        # Print results (keep your existing print statements)
        print("\nValidation Results:")
        print(f"Note Accuracy: {metrics['note_accuracy']:.4f}")
        print(f"Velocity MSE: {metrics['velocity_mse']:.4f}")
        print("\nRhythm Metrics:")
        print(f"Beat Alignment: {metrics['rhythm']['beat_alignment']:.4f}")
        print(f"Density per Beat: {metrics['rhythm']['density_per_beat']:.4f}")
        print("\nDensity Metrics:")
        print(f"Predicted Density: {metrics['density']['pred_density']:.4f}")
        print(f"Target Density: {metrics['density']['target_density']:.4f}")
        print(f"Density Ratio: {metrics['density']['density_ratio']:.4f}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()