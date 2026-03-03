# Medical Term Embeddings

Training transformer models from scratch on synthetic medical knowledge graphs.

## Results

| Model | Data | Category Separation | Relationship (F1) |
|-------|------|---------------------|-------------------|
| **GPT (15 epochs)** | Enhanced | **99.45%** | **67.34%** |
| GPT (10 epochs) | Enhanced | 99.27% | 66.19% |
| GPT | Original | 69.00% | 34.00% |
| BERT | Original | 53.69% | 10.04% |

## Dataset

- **Original**: 6,454 terms, 41,849 relationships, 5 categories
- **Enhanced**: 548 terms, 33,391 relationships, 7 categories

### Categories (Enhanced)
disease, drug, symptom, anatomy, treatment, test, pathogen

### Relation Types (Enhanced)
may_cause, detects, treats, affects, treated_by, causes, has_side_effect, related_to, used_for, associated_with, identifies, interacts_with, has_symptom, caused_by, may_treat, diagnosed_by, isa

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Generate enhanced data
python generate_medical_kg_enhanced.py

# Train GPT embeddings (cosine loss)
python train_gpt_hybrid.py
```

## Files

- `generate_medical_kg.py` - Original data generator
- `generate_medical_kg_enhanced.py` - Enhanced data generator (7 categories)
- `train_gpt_hybrid.py` - GPT training with switchable loss (cosine/transe/hybrid)
- `train_bert_hybrid.py` - BERT training with switchable loss

## Hardware

RTX 3050 (4GB VRAM) - limits model size to 2 layers, 128 dim

## Key Findings

1. GPT outperforms BERT for this task
2. Cosine loss works better than TransE/Hybrid
3. Enhanced data (7 categories) performs significantly better than original (5 categories)
4. More epochs improve marginally (10 → 15)
