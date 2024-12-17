import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from urllib.parse import urlparse
from collections import Counter
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class URLDataset(Dataset):
    def __init__(self, urls, features, labels, tokenizer, max_length=128, selected_features=None):
        self.urls = urls
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selected_features = selected_features

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        features = self.features[idx]
        label = self.labels[idx]
        
        # Tokenize URL
        encoding = self.tokenizer(
            url,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long),
            'urls': url
        }

class URLDataPreprocessor:
    def __init__(self, batch_size=16, debug_sample_size=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.batch_size = batch_size
        self.max_length = 128
        self.scaler = StandardScaler()
        self.debug_sample_size = debug_sample_size
        self.selected_features = None
    
    def extract_features(self, url):
        """Extract numerical features from URL"""
        parsed = urlparse(url)
        return {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'query_length': len(parsed.query),
            'fragment_length': len(parsed.fragment),
            'subdomain_count': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0,
            'special_char_count': sum(c in url for c in '.-_@&=+$,'),
            'digit_count': sum(c.isdigit() for c in url),
            'letter_count': sum(c.isalpha() for c in url),
            'is_https': int(url.startswith('https')),
            'has_port': int(':' in parsed.netloc),
            'has_params': int(len(parsed.query) > 0),
            'dots_in_domain': parsed.netloc.count('.'),
            'hyphen_count': url.count('-'),
            'at_symbol_count': url.count('@'),
            'double_slash_count': url.count('//')
        }

    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Select top features (e.g., top 8)
        top_features = feature_importance.head(8)['feature'].tolist()
        return top_features

    def collate_fn(self, batch):
        """Custom collate function to properly batch the data"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        features = torch.stack([item['features'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        urls = [item['urls'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'features': features,
            'labels': labels,
            'urls': urls
        }

    def prepare_data(self, csv_path):
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Total URLs loaded: {len(df)}")
        
        # Extract features for each URL
        features_list = []
        urls = df['URL'].values
        for url in urls:
            features = self.extract_features(url)
            features_list.append(list(features.values()))
        
        # Convert to numpy array and scale features
        features_array = np.array(features_list)
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Print feature statistics
        print("\nFeature Statistics:")
        print(f"Feature shape: {scaled_features.shape}")
        print("Feature means:", np.mean(scaled_features, axis=0))
        print("Feature stds:", np.std(scaled_features, axis=0))
        
        # Analyze feature importance
        feature_names = list(self.extract_features(urls[0]).keys())
        X = pd.DataFrame(scaled_features, columns=feature_names)
        top_features = self.analyze_feature_importance(X, df['Label'].values)
        
        # Select only top features
        selected_features = X[top_features]
        print("\nSelected features:", top_features)
        
        # Balance dataset
        balanced_data = self.balance_dataset(urls, selected_features, df['Label'].values)
        
        # Add data augmentation
        augmented_data = []
        for url, features, label in zip(urls, selected_features.values, df['Label'].values):
            # Original data
            augmented_data.append((url, features, label))
            
            # URL variations
            if random.random() < 0.3:
                if url.startswith('www.'):
                    augmented_data.append((url.replace('www.', ''), features, label))
                else:
                    augmented_data.append((f'www.{url}', features, label))
                    
            if random.random() < 0.3:
                if url.startswith(('http://', 'https://')):
                    augmented_data.append((url.split('://')[-1], features, label))
                else:
                    augmented_data.append((f'https://{url}', features, label))
        
        # Create and return dataloaders
        return self.create_dataloaders(balanced_data)

    def analyze_urls(self, urls, labels):
        print("\nPhishing URL Analysis:")
        
        def extract_url_features(url):
            parsed = urlparse(url)
            return {
                'tld': parsed.netloc.split('.')[-1],
                'subdomain_count': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0,
                'path_length': len(parsed.path),
                'has_params': len(parsed.query) > 0,
                'special_chars': sum(c in url for c in '.-_@'),
                'has_numbers': any(c.isdigit() for c in parsed.netloc),
                'is_https': url.startswith('https'),
            }
        
        print("\nAnalyzing URL patterns by class:")
        for label, label_name in [(0, "Phishing"), (1, "Legitimate")]:
            class_urls = [url for url, l in zip(urls, labels) if l == label]
            features = [extract_url_features(url) for url in class_urls]
            
            print(f"\n{label_name} URLs characteristics:")
            print(f"Total URLs: {len(class_urls)}")
            print(f"HTTPS usage: {sum(f['is_https'] for f in features) / len(features):.2%}")
            print(f"Contains numbers in domain: {sum(f['has_numbers'] for f in features) / len(features):.2%}")
            print(f"Average subdomains: {np.mean([f['subdomain_count'] for f in features]):.2f}")
            
            # Top TLDs
            tlds = [f['tld'] for f in features]
            print("\nTop 5 TLDs:")
            for tld, count in Counter(tlds).most_common(5):
                print(f"{tld}: {count}")

    def balance_dataset(self, urls, features, labels):
        """Balance the dataset by sampling equal numbers of each class"""
        unique_labels = np.unique(labels)
        min_count = min(np.sum(labels == label) for label in unique_labels)
        target_size = min(self.debug_sample_size or 15000, min_count)
        
        print(f"\nBalancing dataset to {target_size} samples per class")
        balanced_data = {
            'urls': [],
            'features': [],
            'labels': []
        }
        
        for label in unique_labels:
            # Get indices for this class
            indices = np.where(labels == label)[0]
            # Randomly sample indices
            selected_indices = np.random.choice(indices, target_size, replace=False)
            
            # Add selected samples to balanced dataset
            balanced_data['urls'].extend(urls[selected_indices])
            balanced_data['features'].append(features.iloc[selected_indices].values)
            balanced_data['labels'].extend([label] * target_size)
        
        # Convert features to numpy array
        balanced_data['features'] = np.vstack(balanced_data['features'])
        balanced_data['labels'] = np.array(balanced_data['labels'])
        
        print("\nBalanced dataset statistics:")
        print(f"Total samples: {len(balanced_data['labels'])}")
        print("Class distribution:", np.bincount(balanced_data['labels']))
        
        return balanced_data

    def create_dataloaders(self, balanced_data):
        """Create train and validation dataloaders from balanced data"""
        # Split into train and validation
        indices = np.arange(len(balanced_data['labels']))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=0.2,
            stratify=balanced_data['labels'],
            random_state=42
        )
        
        # Create train dataset
        train_dataset = URLDataset(
            urls=np.array(balanced_data['urls'])[train_idx],
            features=balanced_data['features'][train_idx],
            labels=balanced_data['labels'][train_idx],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            selected_features=self.selected_features
        )
        
        # Create validation dataset
        val_dataset = URLDataset(
            urls=np.array(balanced_data['urls'])[val_idx],
            features=balanced_data['features'][val_idx],
            labels=balanced_data['labels'][val_idx],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            selected_features=self.selected_features
        )
        
        print("\nDataloader statistics:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        
        return train_loader, val_loader

    def extract_traditional_features(self, csv_path):
        """Extract traditional features from URLs for ensemble models"""
        df = pd.read_csv(csv_path)
        features = []
        
        print("Extracting traditional features...")
        for url in df['URL']:
            url_features = self.extract_features(url)
            features.append(list(url_features.values()))
        
        # Convert to numpy array
        features = np.array(features)
        feature_names = list(self.extract_features(df['URL'].iloc[0]).keys())
        
        # Analyze and select important features
        if self.selected_features is None:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, self.get_labels(csv_path))
            
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top 8 features
            self.selected_features = importance.head(8)['feature'].tolist()
            print("\nSelected features:", self.selected_features)
        
        # Extract selected features
        feature_indices = [feature_names.index(f) for f in self.selected_features]
        selected_features = features[:, feature_indices]
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(selected_features)
        
        print(f"Extracted {scaled_features.shape[1]} features from {scaled_features.shape[0]} URLs")
        return scaled_features
    
    def get_labels(self, csv_path):
        """Extract labels from dataset"""
        df = pd.read_csv(csv_path)
        return df['Label'].values