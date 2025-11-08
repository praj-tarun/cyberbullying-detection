import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import re
import string

from src.models import LSTMClassifier, SimpleCNN
from src.utils import train_epoch, evaluate

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[{}]".format(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_vocab=10000):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_vocab-2]:
        vocab[word] = len(vocab)
    
    return vocab


def text_to_sequence(text, vocab, max_len=100):
    words = text.split()[:max_len]
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    return sequence


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = text_to_sequence(self.texts[idx], self.vocab, self.max_len)
        return torch.LongTensor(text), torch.FloatTensor(self.labels[idx])


def main():
    # load data
    df = pd.read_csv("data/raw/train.csv")
    df["clean_text"] = df["comment_text"].apply(clean_text)
    
    # get label columns
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values, df[label_cols].values,
        test_size=0.2, random_state=42
    )
    
    # build vocab
    vocab = build_vocab(X_train, max_vocab=10000)
    
    # save vocab and label columns
    with open('outputs/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('outputs/label_cols.pkl', 'wb') as f:
        pickle.dump(label_cols, f)
    
    # create datasets
    train_dataset = TextDataset(X_train, y_train, vocab, max_len=100)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        output_dim=len(label_cols),
        n_layers=2,
        dropout=0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # training
    n_epochs = 10
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'outputs/best_model.pt')
        
        print(f'Epoch: {epoch+1:02d}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


if __name__ == '__main__':
    main()
