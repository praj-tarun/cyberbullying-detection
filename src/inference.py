import torch
import pickle
import re
import string
from src.models import LSTMClassifier, SimpleCNN

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[{}]".format(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_to_sequence(text, vocab, max_len=100):
    words = text.split()[:max_len]
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    return sequence

def load_model(model_type='lstm', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('outputs/label_cols.pkl', 'rb') as f:
        label_cols = pickle.load(f)
    
    if model_type == 'lstm':
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=128,
            output_dim=len(label_cols),
            n_layers=2,
            dropout=0.3
        ).to(device)
        model.load_state_dict(torch.load('outputs/lstm_model.pt', map_location=device))
    elif model_type == 'cnn':
        model = SimpleCNN(
            vocab_size=len(vocab),
            embedding_dim=100,
            n_filters=100,
            filter_sizes=[3, 4, 5],
            output_dim=len(label_cols),
            dropout=0.5
        ).to(device)
        model.load_state_dict(torch.load('outputs/cnn_model.pt', map_location=device))
    else:
        raise ValueError("model_type must be 'lstm' or 'cnn'")
    
    model.eval()
    return model, vocab, label_cols, device

def predict(text, model, vocab, label_cols, device, threshold=0.5):
    cleaned = clean_text(text)
    sequence = text_to_sequence(cleaned, vocab)
    input_tensor = torch.LongTensor([sequence]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0]
        predictions = (probs > threshold).float()
    
    results = {}
    for label, prob, pred in zip(label_cols, probs, predictions):
        results[label] = {
            'probability': prob.item(),
            'predicted': bool(pred.item())
        }
    
    return results

if __name__ == '__main__':
    model, vocab, label_cols, device = load_model('lstm')
    
    test_comments = [
        "You are an idiot and should be banned",
        "Great article, thanks for sharing!",
        "This is complete garbage and you're stupid"
    ]
    
    print("Testing LSTM model:\n")
    for comment in test_comments:
        print(f"Comment: {comment}")
        results = predict(comment, model, vocab, label_cols, device)
        print("Predictions:")
        for label, result in results.items():
            if result['predicted']:
                print(f"  {label}: {result['probability']:.3f}")
        print()
