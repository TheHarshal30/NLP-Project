#!/usr/bin/env python3
"""
Medical Term Embedding Training using BERT with Hybrid Loss
Supports both Cosine Similarity Loss and TransE Loss
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_LAYERS = 2
HIDDEN_SIZE = 128
NUM_HEADS = 2
MAX_LENGTH = 16

# Loss configuration - can switch between 'cosine', 'transe', or 'hybrid'
LOSS_TYPE = "hybrid"  # Options: "cosine", "transe", "hybrid"


class MedicalBERTEmbedder(nn.Module):
    def __init__(self, vocab_size=30000, num_layers=2, hidden_size=128, num_heads=2, 
                 num_categories=5, num_relations=9, use_transe=False):
        super().__init__()
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
        
        self.bert = BertModel(self.config)
        
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        
        # Relation prediction head
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations)
        )
        
        # TransE relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, hidden_size)
        
        self.use_transe = use_transe
        
    def get_embedding(self, input_ids, attention_mask):
        """Get CLS token embedding"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
    
    def get_relation_embedding(self, relation_indices):
        """Get relation embeddings for TransE"""
        return self.relation_embeddings(relation_indices)
    
    def forward(self, head_ids, head_mask, tail_ids, tail_mask, relation_indices=None):
        head_emb = self.get_embedding(head_ids, head_mask)
        tail_emb = self.get_embedding(tail_ids, tail_mask)
        
        # Relation classification
        combined = torch.cat([head_emb, tail_emb, head_emb * tail_emb], dim=1)
        relation_logits = self.relation_predictor(combined)
        
        # TransE relation embeddings
        relation_emb = None
        if self.use_transe and relation_indices is not None:
            relation_emb = self.get_relation_embedding(relation_indices)
        
        return head_emb, tail_emb, relation_logits, relation_emb


class MedicalKGDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        
        self.terms = {t["id"]: t for t in data["terms"]}
        self.relationships = data["relationships"]
        self.categories = data["categories"]
        
        self.term_list = list(self.terms.values())
        self.term_id_to_idx = {t["id"]: i for i, t in enumerate(self.term_list)}
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        print(f"Loaded {len(self.term_list)} terms")
        print(f"Loaded {len(self.relationships)} relationships")
        
    def __len__(self):
        return len(self.relationships) * 2
    
    def __getitem__(self, idx):
        is_positive = idx % 2 == 0
        idx = idx // 2
        
        rel = self.relationships[idx]
        head_term = self.terms[rel["head"]]
        tail_term = self.terms[rel["tail"]]
        
        if is_positive:
            label = 1
        else:
            while True:
                neg_term = random.choice(self.term_list)
                if neg_term["id"] != rel["tail"]:
                    tail_term = neg_term
                    label = 0
                    break
        
        return {
            "head_text": head_term["text"],
            "tail_text": tail_term["text"],
            "relation": rel["relation"],
            "relation_idx": relation_to_idx.get(rel["relation"], 0),
            "label": label,
            "head_category": head_term["category"],
            "tail_category": tail_term["category"]
        }


def compute_cosine_loss(head_emb, tail_emb, labels_for_cosine):
    """Cosine Embedding Loss"""
    criterion_cosine = nn.CosineEmbeddingLoss()
    return criterion_cosine(head_emb, tail_emb, labels_for_cosine)


def compute_transE_loss(head_emb, relation_emb, tail_emb):
    """
    TransE Loss - relationship as translation
    
    head + relation ≈ tail
    So: head + relation - tail should be close to 0
    """
    translation = head_emb + relation_emb - tail_emb
    distance = torch.sum(translation ** 2, dim=1)
    return torch.mean(distance)


def compute_hybrid_loss(head_emb, tail_emb, relation_emb, labels, cosine_weight=1.0, transe_weight=1.0):
    """Combined Cosine + TransE loss"""
    # Cosine loss
    labels_for_cosine = torch.tensor([1.0 if l == 1 else -1.0 for l in labels]).float().to(head_emb.device)
    cosine_loss = compute_cosine_loss(head_emb, tail_emb, labels_for_cosine)
    
    # TransE loss
    transe_loss = compute_transE_loss(head_emb, relation_emb, tail_emb)
    
    # Combined
    total_loss = cosine_weight * cosine_loss + transe_weight * transe_loss
    
    return total_loss, cosine_loss.item(), transe_loss.item()


def train_epoch(model, dataloader, optimizer, tokenizer, category_to_idx, relation_to_idx, device, loss_type="hybrid"):
    model.train()
    total_loss = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    
    loss_type_display = loss_type.upper()
    if loss_type == "hybrid":
        loss_type_display = "COSINE + TransE"
    
    pbar = tqdm(dataloader, desc=f"Training [{loss_type_display}]")
    for batch_idx, batch in enumerate(pbar):
        head_texts = batch["head_text"]
        tail_texts = batch["tail_text"]
        relations = batch["relation"]
        relation_indices = batch["relation_idx"]
        labels = batch["label"]
        head_cats = batch["head_category"]
        
        head_enc = tokenizer(head_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        tail_enc = tokenizer(tail_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        
        head_ids = head_enc["input_ids"].to(device)
        head_mask = head_enc["attention_mask"].to(device)
        tail_ids = tail_enc["input_ids"].to(device)
        tail_mask = tail_enc["attention_mask"].to(device)
        
        rel_indices = torch.tensor(relation_indices).to(device)
        cat_indices = torch.tensor([category_to_idx.get(c, 0) for c in head_cats]).to(device)
        
        optimizer.zero_grad()
        
        head_emb, tail_emb, rel_logits, relation_emb = model(
            head_ids, head_mask, tail_ids, tail_mask, rel_indices
        )
        
        # Relation classification loss
        rel_loss = criterion_ce(rel_logits, rel_indices)
        
        # Category classification loss
        cat_logits = model.category_classifier(head_emb)
        cat_loss = criterion_ce(cat_logits, cat_indices)
        
        # Choose loss type
        if loss_type == "cosine":
            labels_for_cosine = torch.tensor([1.0 if l == 1 else -1.0 for l in labels]).float().to(device)
            link_loss = compute_cosine_loss(head_emb, tail_emb, labels_for_cosine)
            total_loss_batch = rel_loss + link_loss * 3 + cat_loss
            
        elif loss_type == "transe":
            transe_loss = compute_transE_loss(head_emb, relation_emb, tail_emb)
            total_loss_batch = rel_loss + transe_loss * 3 + cat_loss
            
        else:  # hybrid
            hybrid_loss, cos_loss, trans_loss = compute_hybrid_loss(
                head_emb, tail_emb, relation_emb, labels,
                cosine_weight=1.0, transe_weight=3.0
            )
            total_loss_batch = rel_loss + hybrid_loss + cat_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        pbar.set_postfix({"loss": f"{total_loss_batch.item():.4f}"})
        
    return total_loss / len(dataloader)


def extract_embeddings(model, terms, tokenizer, device, batch_size=64):
    model.eval()
    embeddings = {}
    
    term_texts = [t["text"] for t in terms]
    term_ids = [t["id"] for t in terms]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(term_texts), batch_size), desc="Extracting"):
            batch_texts = term_texts[i:i+batch_size]
            batch_ids = term_ids[i:i+batch_size]
            
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            emb = model.get_embedding(input_ids, attention_mask)
            
            for j, tid in enumerate(batch_ids):
                embeddings[tid] = emb[j].cpu().numpy().tolist()
    
    return embeddings


def main():
    global relation_to_idx
    
    print("="*60)
    print("MEDICAL EMBEDDINGS WITH HYBRID LOSS (BERT + TransE)")
    print("="*60)
    print(f"Loss Type: {LOSS_TYPE}")
    
    print("\nLoading data...")
    dataset = MedicalKGDataset("/home/harshal/medical-embeddings/medical_kg.json")
    
    relation_to_idx = {r: i for i, r in enumerate(set([rel["relation"] for rel in dataset.relationships]))}
    
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    print("\nInitializing BERT model with TransE support...")
    use_transe = LOSS_TYPE in ["transe", "hybrid"]
    model = MedicalBERTEmbedder(
        vocab_size=tokenizer.vocab_size,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_categories=len(dataset.categories),
        num_relations=len(relation_to_idx),
        use_transe=use_transe
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    category_to_idx = dataset.category_to_idx
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        avg_loss = train_epoch(
            model, dataloader, optimizer, tokenizer, 
            category_to_idx, relation_to_idx, DEVICE, LOSS_TYPE
        )
        print(f"Average loss: {avg_loss:.4f}")
    
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, dataset.term_list, tokenizer, DEVICE)
    
    # Output filename based on loss type
    if LOSS_TYPE == "cosine":
        output_path = "/home/harshal/medical-embeddings/medical_embeddings_bert_cosine.json"
    elif LOSS_TYPE == "transe":
        output_path = "/home/harshal/medical-embeddings/medical_embeddings_bert_transe.json"
    else:
        output_path = "/home/harshal/medical-embeddings/medical_embeddings_bert_hybrid.json"
    
    output_data = {
        "terms": dataset.term_list,
        "embeddings": embeddings,
        "categories": dataset.categories,
        "config": {
            "model_type": "BERT",
            "loss_type": LOSS_TYPE,
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEmbeddings saved to {output_path}")
    print(f"Total embeddings: {len(embeddings)}")


if __name__ == "__main__":
    relation_to_idx = {}
    main()
