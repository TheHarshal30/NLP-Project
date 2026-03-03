#!/usr/bin/env python3
"""
Medical Term Embedding Training using GPT Architecture
Trains a GPT-style model from scratch on medical KG data
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters - smaller for faster training
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 2e-5
N_LAYERS = 2
N_EMBED = 128
N_HEADS = 2
MAX_LENGTH = 16


class GPT2ConfigModified(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MedicalGPTEmbedder(nn.Module):
    def __init__(self, vocab_size=50257, n_layers=2, n_embed=128, n_heads=2, n_positions=32, 
                 num_categories=5, num_relations=9):
        super().__init__()
        
        self.config = GPT2ConfigModified(
            vocab_size=vocab_size,
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=n_embed,
            n_positions=n_positions,
            use_cache=False,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
        )
        
        self.gpt = GPT2Model(self.config)
        
        self.category_classifier = nn.Linear(n_embed, num_categories)
        
        self.relation_predictor = nn.Sequential(
            nn.Linear(n_embed * 3, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, num_relations)
        )
        
    def get_embedding(self, input_ids, attention_mask=None):
        """
        GPT doesn't have [CLS] token, so we use mean pooling
        """
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
        else:
            mean_embedding = last_hidden.mean(dim=1)
        
        return mean_embedding
    
    def forward(self, head_ids, head_mask, tail_ids, tail_mask):
        head_emb = self.get_embedding(head_ids, head_mask)
        tail_emb = self.get_embedding(tail_ids, tail_mask)
        
        combined = torch.cat([head_emb, tail_emb, head_emb * tail_emb], dim=1)
        relation_logits = self.relation_predictor(combined)
        
        return head_emb, tail_emb, relation_logits


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
            "label": label,
            "head_category": head_term["category"],
            "tail_category": tail_term["category"]
        }


def train_epoch(model, dataloader, optimizer, tokenizer, category_to_idx, relation_to_idx, device):
    model.train()
    total_loss = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_cosine = nn.CosineEmbeddingLoss()
    
    pbar = tqdm(dataloader, desc="Training GPT")
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
        
        labels_for_cosine = torch.tensor([1.0 if l == 1 else -1.0 for l in labels]).float().to(device)
        cat_indices = torch.tensor([category_to_idx.get(c, 0) for c in head_cats]).to(device)
        
        optimizer.zero_grad()
        
        head_emb, tail_emb, rel_logits = model(head_ids, head_mask, tail_ids, tail_mask)
        
        rel_loss = criterion_ce(rel_logits, rel_indices)
        
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
    print("="*60)
    print("MEDICAL EMBEDDINGS WITH GPT ARCHITECTURE")
    print("="*60)
    
    print("\nLoading data...")
    dataset = MedicalKGDataset("/home/harshal/medical-embeddings/medical_kg.json")
    
    relation_to_idx = {r: i for i, r in enumerate(set([rel["relation"] for rel in dataset.relationships]))}
    
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("\nInitializing GPT model...")
    model = MedicalGPTEmbedder(
        vocab_size=tokenizer.vocab_size,
        n_layers=N_LAYERS,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        num_categories=len(dataset.categories),
        num_relations=len(relation_to_idx)
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    category_to_idx = dataset.category_to_idx
    
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
            "model_type": "GPT2",
            "n_layers": N_LAYERS,
            "n_embed": N_EMBED,
            "n_heads": N_HEADS
        }
    }
    
    output_path = "/home/harshal/medical-embeddings/medical_embeddings_gpt.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEmbeddings saved to {output_path}")
    print(f"Total embeddings: {len(embeddings)}")


if __name__ == "__main__":
    main()
