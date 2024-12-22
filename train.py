# Cell 1: Imports
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import URLDataPreprocessor
from model_setup import URLBertClassifier
import torch
import gc
from visualization import ModelVisualizer
import os
from hyperparameter_tuning import HyperparameterOptimizer
from ensemble_classifier import URLEnsembleClassifier
import numpy as np

# Cell 2: Training
def train_model(csv_path, num_epochs=3, optimize_hyperparams=True):
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameter optimization
    if optimize_hyperparams:
        print("\nStarting hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(csv_path, n_trials=20)
        best_params = optimizer.optimize()
        print("\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
    
    # Initialize preprocessor and load data
    batch_size = best_params['batch_size'] if optimize_hyperparams else 32
    preprocessor = URLDataPreprocessor(batch_size=batch_size)
    train_loader, val_loader = preprocessor.prepare_data(csv_path)

    # Initialize model
    if optimize_hyperparams:
        classifier = URLBertClassifier(
            device=device,
            dropout_rate=best_params['dropout_rate'],
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers']
        )
    else:
        classifier = URLBertClassifier(device=device)
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    save_path = 'model_plots'
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize ensemble with cross-validation
    ensemble = URLEnsembleClassifier(classifier, device=device, n_folds=5)
    
    # Extract features and perform cross-validation
    features = preprocessor.extract_traditional_features(csv_path)
    labels = preprocessor.get_labels(csv_path)
    
    # Convert features to torch tensor if needed
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    # Pass the scaler to the ensemble
    ensemble.scaler = preprocessor.scaler
    
    try:
        # Attempt to collect BERT inputs
        print("\nCollecting input_ids and attention_masks from training data...")
        # Process BERT inputs in batches
        bert_probs_list = []
        
        with torch.no_grad():  # Disable gradient computation
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_features = batch['features'].to(device)
                
                # Get BERT predictions for this batch
                bert_outputs = classifier(input_ids, attention_mask, batch_features)
                batch_probs = torch.softmax(bert_outputs, dim=1)
                bert_probs_list.append(batch_probs.cpu().numpy())
                
                # Clear GPU memory after each batch
                del input_ids, attention_mask, batch_features, bert_outputs, batch_probs
                torch.cuda.empty_cache()
        
        # Combine all BERT predictions
        all_bert_probs = np.vstack(bert_probs_list)
        
        # Train with cross-validation and BERT predictions
        ensemble.train_traditional_models(features, labels, bert_probs=all_bert_probs)
    except Exception as e:
        print(f"Warning: Could not collect BERT inputs ({str(e)})")
        print("Falling back to traditional features only")
        # Train with cross-validation using only traditional features
        ensemble.train_traditional_models(features, labels)
    
    # Add ensemble to classifier
    classifier.ensemble = ensemble
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = classifier.train_epoch(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate
        val_loss, val_accuracy, metrics = classifier.evaluate(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(classifier.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # Update visualizer
        visualizer.update_metrics(
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            bert_accuracy=metrics['bert_accuracy'],
            ensemble_accuracy=metrics['ensemble_accuracy'],
            val_precision=metrics['precision'],
            val_recall=metrics['recall'],
            val_f1=metrics['f1']
        )
    
    # Generate plots
    visualizer.plot_training_history(save_path)
    visualizer.plot_confusion_matrix(metrics['true_labels'], metrics['predictions'], save_path)
    visualizer.plot_feature_importance(metrics['feature_names'], metrics['importance_scores'], save_path)
    visualizer.plot_learning_dynamics(save_path)
    
    # Print detailed performance summary
    visualizer.print_performance_summary()

# Cell 3: Run training
dataset_path = 'dataset_50k.csv'
num_epochs = 3
train_model(dataset_path, num_epochs)