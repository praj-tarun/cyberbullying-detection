# Identifying Cyberbullying in Social Media Posts

Deep learning models for multi-label cyberbullying detection using PyTorch. Implements LSTM, CNN, BERT, and Ensemble approaches on the Jigsaw Toxic Comment dataset.

## Dependencies

### Required Packages
- Python 3.8+
- torch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- jupyter >= 1.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- transformers >= 4.30.0 (for BERT)

### Installation
```bash
pip install -r requirements.txt
```

## Dataset

**Cyberbullying Classification**
- Source: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
- 46017 Unique value (social media posts)
- The data has been balanced in order to contain ~8000 of each class.
- Labels: region, age, gender, ethinicity, not_cyberbullying, other_cyberbullying

**Jigsaw Toxic Comment Classification Challenge**
- Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- 160k Wikipedia comments (social media posts)
- Multi-label cyberbullying detection (6 categories)
- Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

### Download Dataset
```bash
# Install kaggle CLI
pip install kaggle

# Download direct from Link(requires kaggle.json in ~/.kaggle/)
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/raw

# Extract files to data/raw/
```

## Project Structure

```
DL/
├── src/                     # Source code
│   ├── __init__.py
│   ├── models.py           # LSTM, CNN, and BERT architectures
│   ├── train.py            # Training script
│   └── utils.py            # Helper functions
├── notebooks/              # Jupyter notebooks
│   ├── Preprocessing.ipynb
│   ├── jigsaw_cnn.ipynb
│   ├── jigsaw_rnn.ipynb
│   ├── jigsaw_lstm.ipynb
│   ├── jigsaw_bert.ipynb
│   ├── cb_tweet_cnn.ipynb
│   ├── cb_tweet_rnn.ipynb
│   ├── cb_tweet_lstm.ipynb
│   ├── cb_tweet_bert.ipynb
│   ├── jigsaw_cnn_colab.ipynb
│   ├── jigsaw_lstm_colab.ipynb
│   └── jigsaw_bert_colab.ipynb
├── outputs/                # Results and artifacts
│   ├── lstm_model.pt
│   ├── cnn_model.pt
│   ├── bert_model.pt
│   ├── vocab.pkl
│   └── label_cols.pkl
├── data/                   # Data directory
│   └── raw/
│       ├── cb_tweets.csv
│       ├── train.csv
│       ├── test.csv
│       └── test_labels.csv
├── README.md
├── PROJECT_REPORT.md
└── requirements.txt
```

## File Descriptions

### Source Code (src/)

**models.py** - Neural network architectures:
- LSTMClassifier: Bidirectional LSTM (2 layers, 128 hidden units)
- SimpleCNN: Multi-filter CNN (filter sizes: 3, 4, 5)
- SimpleRNN: Simple RNN
- BERTClassifier: Fine-tuned BERT for multi-label classification

**train.py** - Training utilities:
- clean_text: Text preprocessing (removes URLs, punctuation, etc.)
- build_vocab: Creates word-to-index mapping (10k words)
- TextDataset: PyTorch Dataset for multi-label data

**utils.py** - Evaluation functions:
- train_epoch: Trains model for one epoch
- evaluate: Evaluates on validation set
- get_predictions: Gets predictions with threshold
- calculate_metrics: Computes precision, recall, F1 per label

**inference.py** - Inference script for testing trained models

### Notebooks (notebooks/)

- **Preprocessing.ipynb**: Data exploration and analysis
- **jigsaw_cnn.ipynb**: CNN implementation on local with 146K data
- **jigsaw_rnn.ipynb**: RNN implementation on local with 146K data
- **jigsaw_lstm.ipynb**: LSTM implementation on local with 146K data
- **jigsaw_bert.ipynb**: BERT implementation on local with 146K data
- **cb_tweet_cnn.ipynb**: CNN implementation on local with 46K data
- **cb_tweet_rnn.ipynb**: RNN implementation on local with 46K data
- **cb_tweet_lstm.ipynb**: LSTM implementation on local with 46K data
- **cb_tweet_bert.ipynb**: BERT implementation on local with 46K data
- **jigsaw_cnn_colab.ipynb**: CNN implementation on colab with 146K data
- **jigsaw_lstm_colab.ipynb**: LSTM implementation on colab with 146K data
- **jigsaw_bert_colab.ipynb**: BERT implementation on colab with 146K data


## Training

### Using Notebooks (Recommended)

Each notebook is self-contained and ready to run:


Trains LSTM model and saves to `outputs/best_model.pt`

### Hyperparameters

- Vocabulary: 10,000 words
- Embedding dim: 100
- Hidden dim: 128 (LSTM), 100 filters (CNN)
- Dropout: 0.3 (LSTM), 0.5 (CNN)
- Batch size: 64 (LSTM/CNN), 16 (BERT)
- Learning rate: 0.001 (LSTM/CNN), 2e-5 (BERT)
- Epochs: 10 (LSTM/CNN), 3 (BERT)
- Max length: 100 (LSTM/CNN), 128 (BERT)

## Inference

### Using Inference Script

```bash
python -m src.inference
```

Tests trained models on sample comments.

### Manual Inference

```python
from src.inference import load_model, predict

# Load model
model, vocab, label_cols, device = load_model('lstm')  # or 'cnn'

# Predict
text = "Your comment here"
results = predict(text, model, vocab, label_cols, device)

for label, result in results.items():
    if result['predicted']:
        print(f"{label}: {result['probability']:.3f}")
```

## Saved Models

After training, models are saved in `outputs/`:
- `lstm_model.pt` - LSTM weights
- `cnn_model.pt` - CNN weights
- `bert_model.pt` - BERT weights
- `vocab.pkl` - Vocabulary mapping
- `label_cols.pkl` - Label names

## Notes

- Multi-label classification: posts can have multiple cyberbullying categories
- Uses BCEWithLogitsLoss for multi-label learning
- GPU automatically used if available
- Class imbalance: threat and identity_hate are rare categories
- BERT trained on 20k subset due to computational constraints
