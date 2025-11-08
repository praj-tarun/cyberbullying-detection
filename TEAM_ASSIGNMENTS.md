# Team Member Assignments

## Member 1: LSTM Model

**Notebook**: `notebooks/lstm_model.ipynb`

**Tasks**:
1. Run notebook completely
2. Record metrics (F1, precision, recall, accuracy, training time)
3. Save model to `outputs/lstm_model.pt`
4. Add results to PROJECT_REPORT.md (Section 6)

**Deliverables**:
- Executed notebook with outputs
- `outputs/lstm_model.pt`
- `outputs/lstm_training.png`
- `outputs/lstm_f1_scores.png`

---

## Member 2: CNN Model

**Notebook**: `notebooks/cnn_model.ipynb`

**Tasks**:
1. Run notebook completely
2. Record metrics (F1, precision, recall, accuracy, training time)
3. Save model to `outputs/cnn_model.pt`
4. Add results to PROJECT_REPORT.md (Section 6)

**Deliverables**:
- Executed notebook with outputs
- `outputs/cnn_model.pt`
- `outputs/cnn_training.png`
- `outputs/cnn_f1_scores.png`

---

## Member 3: Ensemble Model

**Notebook**: `notebooks/ensemble_model.ipynb`

**Prerequisites**: Member 1 and 2 must complete first

**Tasks**:
1. Run notebook after LSTM and CNN models are trained
2. Compare ensemble strategies (average, weighted, max)
3. Record best ensemble F1 score
4. Add results to PROJECT_REPORT.md (Section 6)

**Deliverables**:
- Executed notebook with outputs
- `outputs/ensemble_comparison.png`
- `outputs/ensemble_per_label.png`

---

## Member 4: BERT Model

**Notebook**: `notebooks/bert_model.ipynb`

**Tasks**:
1. Run notebook completely (trains on 20k subset)
2. Record metrics (F1, precision, recall, accuracy, training time)
3. Save model to `outputs/bert_model.pt`
4. Add results to PROJECT_REPORT.md (Section 6)

**Deliverables**:
- Executed notebook with outputs
- `outputs/bert_model.pt`
- `outputs/bert_training.png`
- `outputs/bert_f1_scores.png`

**Note**: Requires ~8GB GPU memory. Reduce batch size if needed.

---

## Shared Tasks

All members:
1. Use same preprocessing (clean_text function)
2. Use same train/test split (random_state=42)
3. Fill in PROJECT_REPORT.md Section 6 with your results
4. Ensure notebooks run without errors

## Execution Order

**Parallel**: Members 1, 2, 4 can work simultaneously

**Sequential**: Member 3 starts after Members 1 & 2 complete

## Final Checklist

- [ ] All 4 notebooks executed with outputs
- [ ] All model files in `outputs/`
- [ ] All plots generated
- [ ] PROJECT_REPORT.md Section 6 completed with all metrics
- [ ] Code runs without errors
