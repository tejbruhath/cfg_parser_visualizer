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

def train_model():
    # Prepare data
    sentences, parse_trees, word_to_idx, tag_to_idx = prepare_data()
    
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
    
    # Prepare training data
    train_data = list(zip(sentences, parse_trees))
    random.shuffle(train_data)
    
    # Training loop
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
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
            
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total
        
        avg_loss = epoch_loss / (len(train_data) / batch_size)
        accuracy = epoch_correct / epoch_total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'static/parser_model.pth')
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('static/training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Train the model
    train_model() 