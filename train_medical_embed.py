#!/usr/bin/env python3
"""
Medical Term Embedding Training
Trains a 4-layer BERT from scratch on medical KG data
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
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_LAYERS = 2
HIDDEN_SIZE = 128
NUM_HEADS = 2
MAX_LENGTH = 16


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
        return len(self.relationships) * 2  #正负样本
    
    def __getitem__(self, idx):
        is_positive = idx % 2 == 0
        idx = idx // 2
        
        rel = self.relationships[idx]
        head_term = self.terms[rel["head"]]
        tail_term = self.terms[rel["tail"]]
        
        if is_positive:
            label = 1
        else:
            # Random negative sample
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
            "label": label,
            "head_category": head_term["category"],
            "tail_category": tail_term["category"]
        }


class MedicalEmbedder(nn.Module):
    def __init__(self, vocab_size=30000, num_layers=4, hidden_size=256, num_heads=4, num_categories=5, num_relations=9):
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
        
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations)
        )
        
        self.contrastive_proj = nn.Linear(hidden_size, hidden_size)
        
    def get_embedding(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
    
    def forward(self, head_ids, head_mask, tail_ids, tail_mask):
        head_emb = self.get_embedding(head_ids, head_mask)
        tail_emb = self.get_embedding(tail_ids, tail_mask)
        
        combined = torch.cat([head_emb, tail_emb, head_emb * tail_emb], dim=1)
        relation_logits = self.relation_predictor(combined)
        
        return head_emb, tail_emb, relation_logits


def train_epoch(model, dataloader, optimizer, tokenizer, category_to_idx, relation_to_idx, device):
    model.train()
    total_loss = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_cosine = nn.CosineEmbeddingLoss()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        head_texts = batch["head_text"]
        tail_texts = batch["tail_text"]
        relations = batch["relation"]
        labels = batch["label"]
        head_cats = batch["head_category"]
        
        head_enc = tokenizer(head_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        tail_enc = tokenizer(tail_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        
        head_ids = head_enc["input_ids"].to(device)
        head_mask = head_enc["attention_mask"].to(device)
        tail_ids = tail_enc["input_ids"].to(device)
        tail_mask = tail_enc["attention_mask"].to(device)
        
        rel_indices = torch.tensor([relation_to_idx.get(r, 0) for r in relations]).to(device)
        labels_tensor = torch.tensor(labels).float().to(device)
        cat_indices = torch.tensor([category_to_idx.get(c, 0) for c in head_cats]).to(device)
        
        optimizer.zero_grad()
        
        head_emb, tail_emb, rel_logits = model(head_ids, head_mask, tail_ids, tail_mask)
        
        rel_loss = criterion_ce(rel_logits, rel_indices)
        
        # Use cosine embedding loss for better relationship learning
        labels_for_cosine = torch.tensor([1.0 if l == 1 else -1.0 for l in labels]).to(device)
        link_loss = criterion_cosine(head_emb, tail_emb, labels_for_cosine)
        
        cat_logits = model.category_classifier(head_emb)
        cat_loss = criterion_ce(cat_logits, cat_indices)
        
        loss = rel_loss + link_loss * 3 + cat_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
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
    print("Loading data...")
    dataset = MedicalKGDataset("/home/harshal/medical-embeddings/medical_kg.json")
    
    relation_to_idx = {rel: i for i, rel in enumerate(dataset.categories)}  # Simplified
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    num_relations = len(dataset.relationships) if len(dataset.relationships) > 0 else 9
    
    print("\nInitializing model...")
    model = MedicalEmbedder(
        vocab_size=tokenizer.vocab_size,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_categories=len(dataset.categories),
        num_relations=9
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    category_to_idx = dataset.category_to_idx
    relation_to_idx = {r: i for i, r in enumerate(set([rel["relation"] for rel in dataset.relationships]))}
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        avg_loss = train_epoch(model, dataloader, optimizer, tokenizer, category_to_idx, relation_to_idx, DEVICE)
        print(f"Average loss: {avg_loss:.4f}")
    
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, dataset.term_list, tokenizer, DEVICE)
    
    output_data = {
        "terms": dataset.term_list,
        "embeddings": embeddings,
        "categories": dataset.categories,
        "config": {
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS
        }
    }
    
    output_path = "/home/harshal/medical-embeddings/medical_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEmbeddings saved to {output_path}")
    print(f"Total embeddings: {len(embeddings)}")
    
    print("\nEvaluating embedding quality...")
    evaluate_embeddings(embeddings, dataset)


def evaluate_embeddings(embeddings, dataset):
    """Evaluate embedding quality"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Group by category
    cats = {}
    for term in dataset.term_list:
        cat = term["category"]
        if cat not in cats:
            cats[cat] = []
        if term["id"] in embeddings:
            cats[cat].append((term["id"], term["text"], np.array(embeddings[term["id"]])))
    
    print("\nCategory clustering quality:")
    for cat, items in cats.items():
        if len(items) < 2:
            continue
        vectors = np.array([item[2] for item in items])
        mean_sim = np.mean([cosine_similarity([v1], [v2])[0,0] for i, v1 in enumerate(vectors) for v2 in vectors[i+1:]])
        print(f"  {cat}: avg intra-cluster similarity = {mean_sim:.4f}")
    
    print("\nSample similarity tests:")
    test_pairs = [
        ("diabetes", "insulin"),
        ("heart disease", "heart"),
        ("cough", "pneumonia"),
        ("headache", "migraine"),
    ]
    
    term_emb = {t["text"]: np.array(embeddings[t["id"]]) for t in dataset.term_list if t["id"] in embeddings}
    
    for t1, t2 in test_pairs:
        if t1 in term_emb and t2 in term_emb:
            sim = cosine_similarity([term_emb[t1]], [term_emb[t2]])[0,0]
            print(f"  '{t1}' vs '{t2}': {sim:.4f}")


if __name__ == "__main__":
    main()
