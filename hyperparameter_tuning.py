import optuna
import numpy as np
from model_setup import URLBertClassifier
from data_preprocessing import URLDataPreprocessor
import torch
import matplotlib.pyplot as plt
import os
from visualization import ModelVisualizer

class HyperparameterOptimizer:
    def __init__(self, csv_path, n_trials=20):
        self.csv_path = csv_path
        self.n_trials = n_trials
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_params = None
        self.trial_scores = []
        
    def objective(self, trial):
        # Hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 384, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        }
        
        # Initialize preprocessor with trial batch size
        preprocessor = URLDataPreprocessor(batch_size=params['batch_size'])
        train_loader, val_loader = preprocessor.prepare_data(self.csv_path)
        
        # Initialize model with trial parameters
        model = URLBertClassifier(
            device=self.device,
            dropout_rate=params['dropout_rate'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers']
        )
        
        # Training setup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Quick evaluation (1 epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = model(input_ids, attention_mask, features)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_f1 = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask, features)
                preds = torch.argmax(outputs, dim=1)
                val_f1 += self.calculate_f1(preds.cpu(), labels.cpu())
        
        val_f1 /= len(val_loader)
        self.trial_scores.append(val_f1)
        
        return val_f1
    
    def calculate_f1(self, preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return f1.item()
    
    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print("\nOptimization History:")
        print(f"Number of trials completed: {len(self.trial_scores)}")
        print(f"Best F1 score: {max(self.trial_scores):.4f}")
        self.plot_optimization_history()
        return study.best_params
    
    def plot_optimization_history(self):
        plt.figure(figsize=(12, 6))
        
        # Plot scores
        trials = range(1, len(self.trial_scores) + 1)
        plt.plot(trials, self.trial_scores, 'b-', marker='o', linewidth=2, markersize=8)
        
        # Add best score point
        best_trial = np.argmax(self.trial_scores)
        plt.plot(best_trial + 1, self.trial_scores[best_trial], 'r*', 
                markersize=15, label=f'Best Score: {self.trial_scores[best_trial]:.4f}')
        
        # Customize plot
        plt.title('Hyperparameter Optimization History', fontsize=14, pad=20)
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel('Validation F1 Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Add score annotations
        for i, score in enumerate(self.trial_scores):
            plt.annotate(f'{score:.4f}', 
                        (trials[i], score),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
        
        plt.tight_layout()
        
        # Save and display
        save_dir = 'model_plots'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/hyperparameter_optimization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show() 