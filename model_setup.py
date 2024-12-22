import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from transformers import DistilBertConfig
from torch.nn import functional as F
from torch.nn import Dropout

class URLBertClassifier(nn.Module):
    def __init__(self, device='cuda', num_features=8, dropout_rate=0.2, hidden_size=384, num_layers=3):
        super(URLBertClassifier, self).__init__()
        self.device = device
        self.num_features = num_features
        
        # BERT config
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = dropout_rate
        config.num_labels = 2
        config.output_hidden_states = True
        
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            config=config
        )
        
        # Feature layer with more moderate dropout
        self.feature_layer = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(384),
            nn.Linear(384, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(768)
        )
        
        # Separate layer for numerical features
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 768),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 2)
        )
        
        self.to(device)
        
        # Optimizer setup with reduced weight decay
        bert_params = {
            'params': self.bert.parameters(), 
            'lr': 2e-5,  # Slightly higher learning rate
            'weight_decay': 0.001  # Reduced weight decay
        }
        other_params = {
            'params': list(self.feature_layer.parameters()) + 
                     list(self.feature_processor.parameters()) + 
                     list(self.classifier.parameters()),
            'lr': 1e-4,
            'weight_decay': 0.001
        }
        
        self.optimizer = AdamW([bert_params, other_params])
        
        # Adjusted scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,  # Reduced warmup steps
            num_training_steps=1000
        )
        
        # Reduced mixup regularization
        self.mixup_alpha = 0.1
        
        # Reduced MC dropout
        self.mc_dropout = nn.Dropout(0.2)
        
        # Early stopping settings
        self.best_val_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
        
        # Reduced noise scale
        self.noise_scale = 0.1

    def forward(self, input_ids, attention_mask, features):
        """Modified forward method to handle cases where BERT inputs are not provided"""
        # Ensure features is a torch tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)
        
        if input_ids is not None and attention_mask is not None:
            # URL text processing
            bert_outputs = self.bert.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Process URL embedding with increased noise
            url_embedding = bert_outputs.last_hidden_state[:, 0, :]
        else:
            # If BERT inputs are not provided, use only features
            url_embedding = self.feature_processor(features)
        
        if self.training:
            noise = torch.randn_like(url_embedding) * self.noise_scale
            url_embedding = url_embedding + noise
        
        # Reduced probability of zeroing out embeddings
        if self.training and random.random() < 0.1:
            url_embedding = torch.zeros_like(url_embedding)
        
        # Process numerical features
        processed_features = self.feature_processor(features)
        
        # Modified feature combination
        if self.training and random.random() < 0.05:
            combined = url_embedding
        else:
            # Weighted combination
            alpha = torch.sigmoid(torch.sum(url_embedding * processed_features, dim=1, keepdim=True))
            combined = alpha * url_embedding + (1 - alpha) * processed_features
        
        # Process through feature layer
        feature_output = self.feature_layer(combined)
        
        # Reduced feature masking probability
        if self.training:
            mask = torch.bernoulli(torch.ones_like(feature_output) * 0.9)
            feature_output = feature_output * mask
        
        # Apply mixup during training
        if self.training:
            batch_size = url_embedding.size(0)
            perm = torch.randperm(batch_size).to(self.device)
            lambda_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            url_embedding = lambda_mix * url_embedding + (1 - lambda_mix) * url_embedding[perm]
            processed_features = lambda_mix * processed_features + (1 - lambda_mix) * processed_features[perm]
        
        # Multiple forward passes during inference
        if not self.training:
            num_samples = 5
            outputs = []
            for _ in range(num_samples):
                out = self.mc_dropout(feature_output)
                outputs.append(self.classifier(out))
            return torch.stack(outputs).mean(0)
        
        return self.classifier(feature_output)

    def train_epoch(self, train_loader):
        self.train()
        total_loss = 0
        
        # 10. Label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.2, 1.0]).to(self.device),
            label_smoothing=0.1  # Add label smoothing
        )
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            features = batch['features'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.forward(input_ids, attention_mask, features)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            if batch_idx % 100 == 0:
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).float().mean()
                print(f"Batch {batch_idx}:")
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {correct:.4f}")
                print(f"Label distribution: {torch.bincount(labels)}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Add gradient noise
            for param in self.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * 0.01
                    param.grad += noise
        
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """Modified evaluation to include ensemble predictions"""
        self.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_urls = []
        all_probs = []
        all_features = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                
                # Store batch results
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_features.extend(features.cpu().numpy())

            # Convert lists to numpy arrays
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            all_features = np.vstack(all_features)

            # Calculate BERT accuracy
            bert_accuracy = np.mean(all_preds == all_labels)
            
            # Get ensemble predictions
            ensemble_preds = self.ensemble.predict(torch.tensor(all_probs).to(self.device), 
                                                all_features)
            
            # Calculate ensemble predictions (fixing the issue here)
            ensemble_pred_labels = np.argmax(ensemble_preds, axis=1)
            ensemble_accuracy = np.mean(ensemble_pred_labels == all_labels)
            
            # Calculate metrics using ensemble predictions
            metrics = self.calculate_metrics(
                predictions=ensemble_pred_labels,
                labels=all_labels,
                pred_probs=ensemble_preds
            )
            
            # Add accuracies to metrics
            metrics['bert_accuracy'] = bert_accuracy
            metrics['ensemble_accuracy'] = ensemble_accuracy
            
            # Update ensemble weights
            self.ensemble.update_weights({
                'bert': {'accuracy': float(bert_accuracy)},
                'rf': {'accuracy': float(ensemble_accuracy)},
                'xgb': {'accuracy': float(ensemble_accuracy)},
                'lgb': {'accuracy': float(ensemble_accuracy)},
                'gb': {'accuracy': float(ensemble_accuracy)},
                'et': {'accuracy': float(ensemble_accuracy)},
                'ada': {'accuracy': float(ensemble_accuracy)},
                'svm': {'accuracy': float(ensemble_accuracy)}
            })
            
            print("\nDetailed metrics:")
            print(f"BERT Accuracy: {bert_accuracy:.4f}")
            print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Confusion Matrix:")
            print(confusion_matrix(all_labels, ensemble_pred_labels))
            
            return total_loss / len(val_loader), ensemble_accuracy, metrics

    def calculate_metrics(self, predictions, labels, pred_probs=None):
        """Calculate metrics ensuring tensors are on CPU"""
        # Convert tensors to CPU if they're on GPU
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        
        # Convert to numpy arrays
        predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_labels': labels,
            'predictions': predictions,
            'pred_probs': pred_probs if pred_probs is not None else np.zeros((len(predictions), 2)),
            'feature_names': ['is_https', 'path_length', 'digit_count', 'url_length', 
                            'letter_count', 'special_char_count', 'domain_length', 'dots_in_domain'],
            'importance_scores': [0.55, 0.23, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02]  # Example scores
        }