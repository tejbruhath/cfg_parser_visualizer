import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import treebank
from nltk.tree import Tree
from collections import defaultdict
import random
import json
import os
import time
import psutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Download required NLTK data
nltk.download('treebank')
nltk.download('punkt')

class SimpleParser(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleParser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(output)
        return predictions

def prepare_data():
    # Extract sentences and parse trees from Penn Treebank
    sentences = treebank.sents()
    parse_trees = treebank.parsed_sents()
    
    # Create vocabulary and tag sets
    word_counts = defaultdict(int)
    tag_set = set()
    
    for tree in parse_trees:
        for pos in tree.pos():
            word, tag = pos
            word_counts[word.lower()] += 1
            tag_set.add(tag)
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(word_counts.keys(), 1)}
    word_to_idx['<unk>'] = 0
    word_to_idx['<pad>'] = len(word_to_idx)
    
    # Create tag to index mapping
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_set)}
    tag_to_idx['<pad>'] = len(tag_to_idx)
    
    # Save mappings for web interface
    with open('static/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    with open('static/tag_to_idx.json', 'w') as f:
        json.dump(tag_to_idx, f)
    
    return sentences, parse_trees, word_to_idx, tag_to_idx

def calculate_metrics(model, dataloader, criterion, device, tag_to_idx):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 2)
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    inference_time = time.time() - start_time
    
    # Calculate per-class metrics
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    target_tags = [idx_to_tag[idx] for idx in all_targets if idx != tag_to_idx['<pad>']]
    predicted_tags = [idx_to_tag[idx] for idx in all_predictions if idx != tag_to_idx['<pad>']]
    
    # Generate confusion matrix
    cm = confusion_matrix(target_tags, predicted_tags, labels=list(tag_to_idx.keys())[:-2])
    
    # Calculate memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'inference_time': inference_time,
        'memory_usage': memory_usage,
        'confusion_matrix': cm,
        'classification_report': classification_report(target_tags, predicted_tags, labels=list(tag_to_idx.keys())[:-2])
    }

def plot_loss(metrics_history):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(metrics_history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss vs Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('static/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy(metrics_history):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('static/accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_usage(metrics_history):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_history['memory_usage'], linewidth=2, color='green')
    plt.title('Memory Usage During Training', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Memory (MB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_inference_time(metrics_history):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_history['inference_time'], linewidth=2, color='purple')
    plt.title('Inference Time per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/inference_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(confusion_mat, labels, tag_to_idx):
    # Convert indices to actual POS tags
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    pos_labels = [idx_to_tag[i] for i in labels]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(confusion_mat, 
                xticklabels=pos_labels,
                yticklabels=pos_labels,
                annot=True, 
                fmt='d',
                cmap='Blues',
                square=True)
    plt.title('POS Tagging Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(metrics_history, tag_to_idx):
    # Generate individual plots
    plot_loss(metrics_history)
    plot_accuracy(metrics_history)
    plot_memory_usage(metrics_history)
    plot_inference_time(metrics_history)
    plot_confusion_matrix(metrics_history['confusion_matrix'][-1], 
                         metrics_history['unique_labels'][-1],
                         tag_to_idx)

def train_model():
    # Prepare data
    sentences, parse_trees, word_to_idx, tag_to_idx = prepare_data()
    
    # Split data into train and validation sets
    data = list(zip(sentences, parse_trees))
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Model parameters
    vocab_size = len(word_to_idx)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = len(tag_to_idx)
    
    # Initialize model
    model = SimpleParser(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_to_idx['<pad>'])
    optimizer = optim.Adam(model.parameters())
    
    # Training parameters
    num_epochs = 10
    batch_size = 32
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'memory_usage': [],
        'inference_time': [],
        'confusion_matrix': [],
        'unique_labels': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch data
            batch_sentences = [[word_to_idx.get(w.lower(), word_to_idx['<unk>']) for w in sent] 
                             for sent, _ in batch]
            batch_tags = [[tag_to_idx[tag] for word, tag in tree.pos()] 
                         for _, tree in batch]
            
            # Pad sequences
            max_len = max(len(s) for s in batch_sentences)
            padded_sentences = [s + [word_to_idx['<pad>']] * (max_len - len(s)) for s in batch_sentences]
            padded_tags = [t + [tag_to_idx['<pad>']] * (max_len - len(t)) for t in batch_tags]
            
            # Convert to tensors
            sentences_tensor = torch.LongTensor(padded_sentences)
            tags_tensor = torch.LongTensor(padded_tags)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sentences_tensor)
            
            # Reshape outputs and tags for loss calculation
            outputs = outputs.view(-1, output_dim)
            tags = tags_tensor.view(-1)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, tags)
            predictions = outputs.argmax(dim=1)
            mask = tags != tag_to_idx['<pad>']
            correct = (predictions[mask] == tags[mask]).sum().item()
            total = mask.sum().item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += correct
            train_total += total
        
        # Calculate training metrics
        avg_train_loss = train_loss / (len(train_data) / batch_size)
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                
                # Prepare batch data
                batch_sentences = [[word_to_idx.get(w.lower(), word_to_idx['<unk>']) for w in sent] 
                                 for sent, _ in batch]
                batch_tags = [[tag_to_idx[tag] for word, tag in tree.pos()] 
                             for _, tree in batch]
                
                # Pad sequences
                max_len = max(len(s) for s in batch_sentences)
                padded_sentences = [s + [word_to_idx['<pad>']] * (max_len - len(s)) for s in batch_sentences]
                padded_tags = [t + [tag_to_idx['<pad>']] * (max_len - len(t)) for t in batch_tags]
                
                # Convert to tensors
                sentences_tensor = torch.LongTensor(padded_sentences)
                tags_tensor = torch.LongTensor(padded_tags)
                
                # Forward pass
                outputs = model(sentences_tensor)
                
                # Reshape outputs and tags for loss calculation
                outputs = outputs.view(-1, output_dim)
                tags = tags_tensor.view(-1)
                
                # Calculate loss and accuracy
                loss = criterion(outputs, tags)
                predictions = outputs.argmax(dim=1)
                mask = tags != tag_to_idx['<pad>']
                correct = (predictions[mask] == tags[mask]).sum().item()
                total = mask.sum().item()
                
                val_loss += loss.item()
                val_correct += correct
                val_total += total
        
        # Calculate validation metrics
        avg_val_loss = val_loss / (len(val_data) / batch_size)
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['train_acc'].append(train_accuracy)
        metrics_history['val_acc'].append(val_accuracy)
        metrics_history['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)
        metrics_history['inference_time'].append(time.time())
        
        # Calculate confusion matrix
        val_tags = []
        pred_tags = []
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                batch_sentences = [[word_to_idx.get(w.lower(), word_to_idx['<unk>']) for w in sent] 
                                 for sent, _ in batch]
                batch_tags = [[tag_to_idx[tag] for word, tag in tree.pos()] 
                             for _, tree in batch]
                
                max_len = max(len(s) for s in batch_sentences)
                padded_sentences = [s + [word_to_idx['<pad>']] * (max_len - len(s)) for s in batch_sentences]
                padded_tags = [t + [tag_to_idx['<pad>']] * (max_len - len(t)) for t in batch_tags]
                
                sentences_tensor = torch.LongTensor(padded_sentences)
                tags_tensor = torch.LongTensor(padded_tags)
                
                outputs = model(sentences_tensor)
                _, predicted = torch.max(outputs.data, 2)
                
                for j in range(len(batch)):
                    for k in range(len(batch_tags[j])):
                        if batch_tags[j][k] != tag_to_idx['<pad>']:
                            val_tags.append(batch_tags[j][k])
                            pred_tags.append(predicted[j][k].item())
        
        # Get unique labels that are actually present in the data
        unique_labels = sorted(list(set(val_tags) | set(pred_tags)))
        metrics_history['confusion_matrix'].append(confusion_matrix(
            val_tags,
            pred_tags,
            labels=unique_labels
        ))
        metrics_history['unique_labels'].append(unique_labels)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Plot all metrics
    plot_metrics(metrics_history, tag_to_idx)
    
    # Save metrics history
    with open('static/metrics_history.json', 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (np.float32, np.float64)) 
                      else x.tolist() if isinstance(x, np.ndarray) 
                      else x for x in v] 
                  for k, v in metrics_history.items()}, f)

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Train the model
    train_model() 