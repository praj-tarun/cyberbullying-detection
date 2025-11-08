import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        
        predictions = model(text)
        loss = criterion(predictions, labels)
        acc = accuracy_multilabel(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = accuracy_multilabel(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def accuracy_multilabel(preds, y, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    correct = (preds == y).float()
    return correct.mean()


def get_predictions(model, iterator, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)
            predictions = model(text)
            preds = (torch.sigmoid(predictions) > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true, y_pred, label_names):
    results = {}
    for i, label in enumerate(label_names):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        
        results[label] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    results['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_recall_fscore_support(y_true, y_pred, average='micro')[0],
        'recall': precision_recall_fscore_support(y_true, y_pred, average='micro')[1],
        'f1': precision_recall_fscore_support(y_true, y_pred, average='micro')[2]
    }
    
    return results
