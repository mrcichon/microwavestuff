import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import glob
import os
import re
from typing import Tuple, List, Optional

class SParamWaterDataset(Dataset):
    def __init__(self, data_dirs: List[str], transform_phase: bool = True, augment: bool = False, noise_std: float = 0.01):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        self.data_dirs = data_dirs
        self.transform_phase = transform_phase
        self.augment = augment
        self.noise_std = noise_std
        self.samples = []
        self.targets = []  
        self.distances = []
        self.filenames = []
        self.n_freq = None
        self.n_sparams = 4
        
        self._load_data()
        self._normalize_features()
        
    def _parse_filename(self, filename: str) -> Tuple[float, float]:
        water_match = re.search(r'(\d+)ml', filename.lower())
        
        dist_mm_match = re.search(r'(\d+)mm', filename.lower())
        dist_cm_match = re.search(r'(\d+)cm', filename.lower())
        
        if not water_match:
            raise ValueError(f"Could not parse water volume from filename: {filename}")
        
        if not dist_mm_match and not dist_cm_match:
            raise ValueError(f"Could not parse distance (mm or cm) from filename: {filename}")
            
        water_ml = float(water_match.group(1))
        
        if dist_mm_match:
            distance_mm = float(dist_mm_match.group(1))
        else:
            distance_cm = float(dist_cm_match.group(1))
            distance_mm = distance_cm * 10.0
            
        return water_ml, distance_mm
        
    def _load_s2p_file(self, filepath: str) -> np.ndarray:
        data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('!') or line.startswith('#'):
                    continue
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 9:
                    row = [float(x) for x in parts[:9]]
                    data.append(row)
                    
        return np.array(data)
    
    def _unwrap_phase(self, phase_deg: np.ndarray) -> np.ndarray:
        phase_rad = np.radians(phase_deg)
        unwrapped_rad = np.unwrap(phase_rad)
        return np.degrees(unwrapped_rad)
    
    def _detrend_phase(self, phase_deg: np.ndarray) -> np.ndarray:
        x = np.arange(len(phase_deg))
        coeffs = np.polyfit(x, phase_deg, 1)
        linear_trend = np.polyval(coeffs, x)
        return phase_deg - linear_trend
    
    def _find_s2p_files_recursive(self, data_dirs: List[str]) -> List[str]:
        s2p_files = []
        
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory {data_dir} does not exist")
                continue
                
            print(f"Searching for .s2p files in {data_dir} (recursive)...")
            
            pattern = os.path.join(data_dir, "**", "*.s2p")
            found_files = glob.glob(pattern, recursive=True)
            
            print(f"Found {len(found_files)} .s2p files in {data_dir}")
            s2p_files.extend(found_files)
        
        return s2p_files
        
    def _load_data(self):
        s2p_files = self._find_s2p_files_recursive(self.data_dirs)
        
        if not s2p_files:
            raise ValueError(f"No .s2p files found in directories: {self.data_dirs}")
            
        print(f"\nLoading {len(s2p_files)} S2P files total...")
        
        loaded_count = 0
        skipped_count = 0
        distance_units_found = {'mm': 0, 'cm': 0}
        
        for filepath in s2p_files:
            try:
                filename = os.path.basename(filepath)
                directory = os.path.dirname(filepath)
                
                water_ml, distance_mm = self._parse_filename(filename)
                
                if 'mm' in filename.lower():
                    distance_units_found['mm'] += 1
                elif 'cm' in filename.lower():
                    distance_units_found['cm'] += 1
                
                s_data = self._load_s2p_file(filepath)
                
                if len(s_data) == 0:
                    print(f"Warning: Empty file {filepath}")
                    skipped_count += 1
                    continue
                
                if self.n_freq is None:
                    self.n_freq = len(s_data)
                    print(f"Detected {self.n_freq} frequency points")
                
                if len(s_data) != self.n_freq:
                    print(f"Warning: {filepath} has {len(s_data)} freq points, expected {self.n_freq}")
                    skipped_count += 1
                    continue
                
                frequencies = s_data[:, 0]
                s_params = s_data[:, 1:]
                
                if self.transform_phase:
                    for i in [1, 3, 5, 7]:
                        s_params[:, i] = self._unwrap_phase(s_params[:, i])
                        s_params[:, i] = self._detrend_phase(s_params[:, i])
                
                features = s_params.flatten()
                
                self.samples.append(features)
                self.targets.append(water_ml)
                self.distances.append(distance_mm)
                self.filenames.append(f"{os.path.basename(directory)}/{filename}")
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                skipped_count += 1
                continue
                
        if not self.samples:
            raise ValueError("No valid samples loaded")
        
        print(f"\nDataset Summary:")
        print(f"  Successfully loaded: {loaded_count} samples")
        print(f"  Skipped/failed: {skipped_count} samples")
        print(f"  Distance units found: {distance_units_found['mm']} files with mm, {distance_units_found['cm']} files with cm")
        print(f"  Feature dimension: {len(self.samples[0])}")
        print(f"  Expected: {self.n_freq} freq × {self.n_sparams * 2} params = {self.n_freq * self.n_sparams * 2}")
        print(f"  Water range: {min(self.targets):.0f}ml - {max(self.targets):.0f}ml")
        print(f"  Distance range: {min(self.distances):.0f}mm - {max(self.distances):.0f}mm ({min(self.distances)/10:.1f}cm - {max(self.distances)/10:.1f}cm)")
        
        dir_counts = {}
        for filename in self.filenames:
            dir_name = filename.split('/')[0]
            dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
        
        print(f"  Samples by directory:")
        for dir_name, count in dir_counts.items():
            print(f"    {dir_name}: {count} samples")

    def _normalize_features(self):
        X = np.array(self.samples)
        distances = np.array(self.distances).reshape(-1, 1)
        
        self.scaler_sparams = StandardScaler()
        X_norm = self.scaler_sparams.fit_transform(X)
        
        self.scaler_distance = StandardScaler()
        distances_norm = self.scaler_distance.fit_transform(distances).flatten()
        
        self.samples = [X_norm[i] for i in range(len(X_norm))]
        self.distances = distances_norm.tolist()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.samples[idx])
        distance = torch.FloatTensor([self.distances[idx]])
        target = torch.FloatTensor([self.targets[idx]])
        
        if self.augment:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        
        return features, distance, target

class AntennaAgnosticWaterPredictor(nn.Module):
    def __init__(self, n_freq: int, n_sparams: int = 4, hidden_dim: int = 256, latent_dim: int = 64, 
                 dropout: float = 0.3, use_batchnorm: bool = True):
        super().__init__()
        
        self.n_freq = n_freq
        self.n_sparams = n_sparams
        
        input_channels = n_sparams * 2
        
        conv_dropout = dropout * 0.5
        
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=21, stride=2, padding=10),
            nn.BatchNorm1d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            
            nn.Conv1d(64, 32, kernel_size=15, stride=3, padding=7),  
            nn.BatchNorm1d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            
            nn.Conv1d(32, 16, kernel_size=10, stride=5, padding=4),
            nn.BatchNorm1d(16) if use_batchnorm else nn.Identity(), 
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            
            nn.AdaptiveAvgPool1d(20)
        )
        
        freq_features_dim = 16 * 20
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(freq_features_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batchnorm else nn.Identity(), 
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        self.water_predictor = nn.Linear(latent_dim, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, s_params, distance):
        batch_size = s_params.shape[0]
        
        s_params_reshaped = s_params.view(batch_size, self.n_freq, self.n_sparams * 2)
        s_params_reshaped = s_params_reshaped.transpose(1, 2)
        
        freq_features = self.freq_encoder(s_params_reshaped)
        freq_features = freq_features.view(batch_size, -1)
        
        combined = torch.cat([freq_features, distance], dim=1)
        
        fused_features = self.feature_fusion(combined)
        
        water_volume = self.water_predictor(fused_features)
        
        return water_volume

class WaterContentTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for s_params, distance, target in train_loader:
            s_params = s_params.to(self.device)
            distance = distance.to(self.device) 
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            pred = self.model(s_params, distance)
            loss = criterion(pred, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for s_params, distance, target in val_loader:
                s_params = s_params.to(self.device)
                distance = distance.to(self.device)
                target = target.to(self.device)
                
                pred = self.model(s_params, distance)
                loss = criterion(pred, target)
                
                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
                
        avg_loss = total_loss / len(val_loader)
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        
        return avg_loss, mae, predictions, targets
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=15, 
              weight_decay=0.01, loss_fn='huber', scheduler_patience=5):
        
        if loss_fn == 'huber':
            criterion = nn.HuberLoss(delta=10.0)
        elif loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = nn.L1Loss()
        elif loss_fn == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.HuberLoss(delta=10.0)
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_mae, _, _ = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val MAE {val_mae:.2f}ml")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return best_val_loss

def cross_validate_water_predictor(data_dirs: List[str], n_folds: int = 5, device: str = 'cpu',
                                  hidden_dim: int = 256, latent_dim: int = 64, dropout: float = 0.3,
                                  lr: float = 0.001, weight_decay: float = 0.01, loss_fn: str = 'huber',
                                  augment: bool = False, noise_std: float = 0.01, use_batchnorm: bool = True,
                                  epochs: int = 150, patience: int = 15):
    
    dataset = SParamWaterDataset(data_dirs, augment=augment, noise_std=noise_std)
    
    sample_groups = {}
    for i, (water, distance, filename) in enumerate(zip(dataset.targets, dataset.distances, dataset.filenames)):
        key = (water, distance)
        if key not in sample_groups:
            sample_groups[key] = []
        sample_groups[key].append(i)
    
    group_keys = list(sample_groups.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_groups, val_groups) in enumerate(kf.split(group_keys)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        train_indices = []
        val_indices = []
        
        for group_idx in train_groups:
            train_indices.extend(sample_groups[group_keys[group_idx]])
        for group_idx in val_groups:
            val_indices.extend(sample_groups[group_keys[group_idx]])
            
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        model = AntennaAgnosticWaterPredictor(
            n_freq=dataset.n_freq, 
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )
        trainer = WaterContentTrainer(model, device)
        
        best_val_loss = trainer.train(
            train_loader, val_loader, 
            epochs=epochs, lr=lr, patience=patience,
            weight_decay=weight_decay, loss_fn=loss_fn
        )
        
        final_loss, final_mae, predictions, targets = trainer.validate(val_loader, nn.HuberLoss(delta=10.0))
        
        fold_results.append({
            'val_loss': final_loss,
            'val_mae': final_mae,
            'predictions': predictions,
            'targets': targets,
            'model_state': model.state_dict().copy()
        })
        
        print(f"Fold {fold + 1} Results: Loss {final_loss:.4f}, MAE {final_mae:.2f}ml")
    
    avg_mae = np.mean([r['val_mae'] for r in fold_results])
    std_mae = np.std([r['val_mae'] for r in fold_results])
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Average MAE: {avg_mae:.2f} ± {std_mae:.2f} ml")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    all_predictions = np.concatenate([r['predictions'] for r in fold_results])
    all_targets = np.concatenate([r['targets'] for r in fold_results])
    
    plt.scatter(all_targets, all_predictions, alpha=0.6)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('True Water Volume (ml)')
    plt.ylabel('Predicted Water Volume (ml)')
    plt.title(f'Water Volume Prediction\nMAE: {avg_mae:.1f}±{std_mae:.1f} ml')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    mae_per_fold = [r['val_mae'] for r in fold_results]
    plt.bar(range(1, n_folds + 1), mae_per_fold)
    plt.xlabel('Fold')
    plt.ylabel('MAE (ml)')
    plt.title('MAE by Fold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('water_prediction_results.png', dpi=300)
    plt.show()
    
    return fold_results, avg_mae, std_mae

def train_final_model(data_dirs: List[str], model_save_path: str = 'water_predictor.pth', device: str = 'cpu',
                     hidden_dim: int = 256, latent_dim: int = 64, dropout: float = 0.3,
                     lr: float = 0.001, weight_decay: float = 0.01, loss_fn: str = 'huber',
                     augment: bool = False, noise_std: float = 0.01, use_batchnorm: bool = True,
                     epochs: int = 200, patience: int = 25):
    
    dataset = SParamWaterDataset(data_dirs, augment=augment, noise_std=noise_std)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = AntennaAgnosticWaterPredictor(
        n_freq=dataset.n_freq,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    )
    trainer = WaterContentTrainer(model, device)
    
    print("Training final model...")
    best_val_loss = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, lr=lr, patience=patience,
        weight_decay=weight_decay, loss_fn=loss_fn
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_sparams': dataset.scaler_sparams,
        'scaler_distance': dataset.scaler_distance,
        'model_config': {
            'n_freq': dataset.n_freq,
            'n_sparams': 4,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'dropout': dropout,
            'use_batchnorm': use_batchnorm
        }
    }, model_save_path)
    
    print(f"Final model saved to {model_save_path}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    final_loss, final_mae, predictions, targets = trainer.validate(val_loader, nn.HuberLoss(delta=10.0))
    
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Water Volume (ml)')
    plt.ylabel('Predicted Water Volume (ml)')
    plt.title(f'Final Model Performance\nMAE: {final_mae:.1f} ml')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_model_training.png', dpi=300)
    plt.show()
    
    return model, dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train antenna-agnostic water content predictor")
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, 
                       help='One or more directories containing .s2p files (searched recursively)')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--train_final', action='store_true', help='Train final model')
    parser.add_argument('--model_path', type=str, default='water_predictor.pth', help='Path to save model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['huber', 'mse', 'mae', 'smooth_l1'], 
                       help='Loss function')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Noise standard deviation for augmentation')
    parser.add_argument('--no_batchnorm', action='store_true', help='Disable batch normalization')
    
    args = parser.parse_args()
    
    if args.cv:
        print("Running cross-validation...")
        fold_results, avg_mae, std_mae = cross_validate_water_predictor(
            args.data_dirs, device=args.device,
            hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay, loss_fn=args.loss_fn,
            augment=args.augment, noise_std=args.noise_std, use_batchnorm=not args.no_batchnorm,
            epochs=args.epochs, patience=args.patience
        )
        
    if args.train_final:
        print("Training final model...")
        model, dataset = train_final_model(
            args.data_dirs, args.model_path, device=args.device,
            hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay, loss_fn=args.loss_fn,
            augment=args.augment, noise_std=args.noise_std, use_batchnorm=not args.no_batchnorm,
            epochs=args.epochs, patience=args.patience
        )
