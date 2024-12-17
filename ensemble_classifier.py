import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

class URLEnsembleClassifier:
    def __init__(self, bert_model, device='cuda', n_folds=5):
        self.bert_model = bert_model
        self.device = device
        self.n_folds = n_folds
        self.cv_results = {}
        self.scaler = None
        
        # Initialize traditional models with optimized parameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1.0
        )
        
        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )
        
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.et_classifier = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
        
        self.ada_classifier = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            algorithm='SAMME'
        )
        
        self.svm_classifier = SVC(
            probability=True,
            kernel='rbf',
            random_state=42
        )
        
        # Meta-learner for dynamic weighting
        self.meta_learner = LogisticRegression(max_iter=1000)
        
        # Initialize models dictionary
        self.models = {
            'bert': self.bert_model,
            'rf': self.rf_classifier,
            'xgb': self.xgb_classifier,
            'lgb': self.lgb_classifier,
            'gb': self.gb_classifier,
            'et': self.et_classifier,
            'ada': self.ada_classifier,
            'svm': self.svm_classifier
        }
        
        # Equal initial weights
        n_models = len(self.models)
        self.model_weights = {model: 1.0/n_models for model in self.models}
        
        # Track model performance
        self.model_performance = {model: [] for model in self.models}
        
        # Track feature importance
        self.feature_importance = {}
    
    def cross_validate_models(self, features, labels):
        """Perform k-fold cross validation for all models"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = {model: [] for model in self.models if model != 'bert'}
        
        print(f"\nPerforming {self.n_folds}-fold cross validation:")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Train and evaluate each model
            fold_preds = {}
            for name, model in self.models.items():
                if name != 'bert':  # Skip BERT as it's handled separately
                    print(f"Training {name}...")
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    cv_scores[name].append(score)
                    
                    # Store predictions for ensemble analysis
                    fold_preds[name] = model.predict_proba(X_val)
            
            # Analyze ensemble performance for this fold
            self._analyze_fold_ensemble(fold_preds, y_val, fold)
        
        # Summarize cross-validation results
        self._summarize_cv_results(cv_scores)
    
    def _analyze_fold_ensemble(self, fold_preds, y_true, fold):
        """Analyze ensemble performance for a single fold"""
        # Try different combination weights
        weight_schemes = {
            'equal': {model: 1.0/len(fold_preds) for model in fold_preds},
            'weighted': self.model_weights,
            'confidence': self._calculate_confidence_weights(fold_preds)
        }
        
        for scheme_name, weights in weight_schemes.items():
            # Combine predictions using current weight scheme
            ensemble_pred = sum(
                weights[model] * preds 
                for model, preds in fold_preds.items()
            )
            
            # Calculate metrics
            accuracy = np.mean(np.argmax(ensemble_pred, axis=1) == y_true)
            
            # Store results
            if scheme_name not in self.cv_results:
                self.cv_results[scheme_name] = []
            self.cv_results[scheme_name].append(accuracy)
    
    def _calculate_confidence_weights(self, predictions):
        """Calculate weights based on prediction confidence"""
        confidences = {}
        for name, preds in predictions.items():
            confidences[name] = np.mean(np.max(preds, axis=1))
        
        # Normalize confidences to weights
        total_conf = sum(confidences.values())
        return {name: conf/total_conf for name, conf in confidences.items()}
    
    def _summarize_cv_results(self, cv_scores):
        """Print summary of cross-validation results"""
        print("\nCross-validation Results:")
        
        # Individual model results
        results_df = pd.DataFrame(cv_scores)
        print("\nIndividual Model Performance:")
        print(f"Mean Accuracy (std):")
        for model in results_df.columns:
            mean = results_df[model].mean()
            std = results_df[model].std()
            print(f"{model}: {mean:.4f} (±{std:.4f})")
        
        # Ensemble results
        print("\nEnsemble Performance:")
        for scheme, scores in self.cv_results.items():
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"{scheme} weights: {mean:.4f} (±{std:.4f})")
    
    def train_traditional_models(self, features, labels, bert_probs=None):
        """Modified to include cross-validation and BERT predictions"""
        # Convert features to numpy if it's a tensor
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Perform cross-validation first
        self.cross_validate_models(features, labels)
        
        # Then proceed with regular training
        print("\nTraining final models on full dataset:")
        train_preds = []
        
        for name, model in self.models.items():
            if name != 'bert':
                print(f"Training {name}...")
                model.fit(features, labels)
                pred = model.predict_proba(features)
                train_preds.append(pred)
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
        
        print("Obtaining BERT model predictions for meta-learner...")
        if bert_probs is None:
            # Fallback to using only features if BERT inputs are not provided
            print("Warning: BERT inputs not provided, using features only")
            bert_probs = np.zeros((len(features), 2))
        
        train_preds.append(bert_probs)  # Include BERT's predictions
        
        meta_features = np.hstack(train_preds)
        self.meta_learner.fit(meta_features, labels)
        
        self._print_feature_importance()
    
    def _print_feature_importance(self):
        """Print aggregated feature importance"""
        if self.feature_importance:
            print("\nFeature Importance Summary:")
            avg_importance = np.zeros_like(next(iter(self.feature_importance.values())))
            for imp in self.feature_importance.values():
                avg_importance += imp
            avg_importance /= len(self.feature_importance)
            
            # Sort and print top features
            for idx in np.argsort(avg_importance)[::-1][:10]:
                print(f"Feature {idx}: {avg_importance[idx]:.4f}")
    
    def update_weights(self, val_metrics):
        # Since we're using a meta-learner, manual weight updates may not be necessary.
        pass  # Temporarily disable to prevent interference
    
    def predict(self, bert_logits, features):
        """Combine predictions from all models using the meta-learner"""
        # Get BERT predictions
        bert_probs = bert_logits.cpu().numpy()
        
        # Get predictions from all traditional models
        model_preds = []
        for name, model in self.models.items():
            if name != 'bert':
                try:
                    preds = model.predict_proba(features)
                    model_preds.append(preds)
                except Exception as e:
                    print(f"Error getting predictions from {name}: {e}")
                    continue  # Skip failed models
        
        # Add BERT's predictions
        model_preds.append(bert_probs)
        
        # Concatenate all predictions as meta-features
        if model_preds:
            meta_features = np.hstack(model_preds)
        else:
            print("No model predictions available.")
            return bert_probs  # Fallback to BERT if no other predictions

        # Use meta-learner to predict probabilities
        if hasattr(self, 'meta_learner') and self.meta_learner:
            ensemble_pred = self.meta_learner.predict_proba(meta_features)
        else:
            # Fallback to weighted average if meta_learner is not available
            ensemble_pred = sum(
                self.model_weights[model] * preds 
                for model, preds in zip(self.models.keys(), model_preds)
                if model in self.model_weights
            )
            ensemble_pred /= sum(self.model_weights[model] for model in self.model_weights if model in self.models)
        
        return ensemble_pred

    def print_meta_learner_coefficients(self):
        """Print meta-learner coefficients"""
        print("\nMeta-Learner Coefficients:")
        feature_names = list(self.models.keys())  # ['bert', 'rf', 'xgb', ...]
        coefficients = self.meta_learner.coef_[0]
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")