#!/usr/bin/env python3
"""
Enhanced Medical Knowledge Graph Generator
Creates comprehensive synthetic medical data for embedding training
"""

import json
import random
import os

random.seed(42)

# ============ MEDICAL TERMS DATABASE ============

CATEGORIES = ["disease", "drug", "symptom", "anatomy", "treatment", "test", "pathogen"]

# Comprehensive medical terms
MEDICAL_TERMS = {
    "disease": [
        # Cardiovascular
        "hypertension", "heart failure", "arrhythmia", "cardiomyopathy", "angina", "myocardial infarction",
        "congestive heart failure", "atrial fibrillation", "ventricular tachycardia", "bradycardia",
        "atherosclerosis", "coronary artery disease", "endocarditis", "pericarditis", "aortic aneurysm",
        
        # Respiratory
        "asthma", "copd", "pneumonia", "tuberculosis", "bronchitis", "emphysema", "pulmonary fibrosis",
        "lung cancer", "pulmonary embolism", "pleurisy", "bronchiolitis", "respiratory failure",
        
        # Metabolic
        "diabetes type 1", "diabetes type 2", "metabolic syndrome", "hypothyroidism", "hyperthyroidism",
        "cushing syndrome", "addison disease", "acromegaly", "hyperlipidemia", "hypercholesterolemia",
        
        # Neurological
        "alzheimer disease", "parkinson disease", "epilepsy", "multiple sclerosis", "stroke", "migraine",
        "dementia", "neuropathy", "meningitis", "encephalitis", "brain tumor", "ALS", "huntington disease",
        
        # Gastrointestinal
        "gastritis", "peptic ulcer", "gastroesophageal reflux", "hepatitis", "cirrhosis", "pancreatitis",
        "cholecystitis", "appendicitis", "crohn disease", "ulcerative colitis", "irritable bowel syndrome",
        
        # Renal
        "chronic kidney disease", "kidney failure", "nephritis", "kidney stones", "glomerulonephritis",
        "pyelonephritis", "renal cyst", "urinary tract infection",
        
        # Infectious
        "hiv", "aids", "influenza", "measles", "mumps", "rubella", "chickenpox", "herpes", "herpes zoster",
        "malaria", "dengue", "typhoid", "cholera", "sepsis", "meningitis",
        
        # Autoimmune
        "rheumatoid arthritis", "systemic lupus erythematosus", "scleroderma", "fibromyalgia",
        "grave disease", "hashimoto thyroiditis", "crohn disease", "multiple sclerosis",
        
        # Cancer
        "breast cancer", "lung cancer", "colon cancer", "prostate cancer", "skin cancer", "leukemia",
        "lymphoma", "pancreatic cancer", "ovarian cancer", "brain cancer", "thyroid cancer",
        "bladder cancer", "kidney cancer", "liver cancer", "stomach cancer", "esophageal cancer",
        
        # Psychiatric
        "depression", "anxiety disorder", "bipolar disorder", "schizophrenia", "ptsd", "ocd",
        "panic disorder", "social anxiety", "adhd", "autism spectrum disorder", "eating disorder",
        
        # Musculoskeletal
        "osteoporosis", "osteoarthritis", "rheumatoid arthritis", "fibromyalgia", "gout", "back pain",
        "sciatica", "carpal tunnel", "tendinitis", "bursitis",
        
        # Other
        "anemia", "hemophilia", "leukopenia", "thrombocytopenia", "sickle cell disease",
        "eczema", "psoriasis", "acne", "urtricaria", "cellulitis",
    ],
    
    "drug": [
        # Cardiovascular
        "lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "amlodipine", "diltiazem",
        "metoprolol", "atenolol", "carvedilol", "warfarin", "heparin", "aspirin", "clopidogrel",
        "atorvastatin", "simvastatin", "rosuvastatin", "digoxin", "amiodarone",
        
        # Diabetes
        "metformin", "glipizide", "glibenclamide", "insulin", "liraglutide", "sitagliptin",
        "empagliflozin", "dapagliflozin", "pioglitazone",
        
        # Pain/Anti-inflammatory
        "ibuprofen", "naproxen", "diclofenac", "acetaminophen", "aspirin", "tramadol", "morphine",
        "codeine", "fentanyl", "oxycodone", "hydrocodone",
        
        # Antibiotics
        "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "metronidazole", "cephalexin",
        "ceftriaxone", "vancomycin", "clindamycin", "trimethoprim", "sulfamethoxazole",
        
        # Antivirals
        "oseltamivir", "acyclovir", "valacyclovir", "ribavirin", "remdesivir", "molnupiravir",
        
        # Psychiatric
        "sertraline", "fluoxetine", "escitalopram", "paroxetine", "venlafaxine", "duloxetine",
        "alprazolam", "diazepam", "lorazepam", "clonazepam", "zolpidem",
        "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole",
        
        # Respiratory
        "albuterol", "salbutamol", "fluticasone", "budesonide", "montelukast", "theophylline",
        "ipratropium", "tiotropium",
        
        # GI
        "omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "famotidine",
        "ondansetron", "metoclopramide", "loperamide", "lactulose", "bisacodyl",
        
        # Thyroid
        "levothyroxine", "methimazole", "propylthiouracil",
        
        # Chemotherapy
        "cisplatin", "paclitaxel", "doxorubicin", "cyclophosphamide", "methotrexate",
        "imatinib", "rituximab", "trastuzumab", " Pembrolizumab",
        
        # Other
        "prednisone", "dexamethasone", "hydrocortisone", "gabapentin", "pregabalin",
        "sildenafil", "tadalafil", "allopurinol", "colchicine", "hydroxychloroquine",
    ],
    
    "symptom": [
        # General
        "fever", "fatigue", "weakness", "weight loss", "weight gain", "night sweats", "chills",
        "malaise", "lethargy", "dizziness", "lightheadedness", "syncope",
        
        # Pain
        "headache", "chest pain", "abdominal pain", "back pain", "joint pain", "muscle pain",
        "nerve pain", "burning sensation", "tingling", "numbness", "sharp pain", "dull ache",
        
        # Respiratory
        "cough", "shortness of breath", "wheezing", "sputum production", "hemoptysis",
        "sore throat", "congestion", "runny nose", "sneezing", "hoarseness",
        
        # Cardiovascular
        "palpitations", "rapid heartbeat", "slow heartbeat", "irregular heartbeat", "edema",
        "swelling", "leg swelling",
        
        # GI
        "nausea", "vomiting", "diarrhea", "constipation", "bloating", "abdominal distension",
        "loss of appetite", "heartburn", "difficulty swallowing", "blood in stool",
        
        # Neurological
        "confusion", "memory loss", "difficulty concentrating", "slurred speech", "seizure",
        "tremor", "balance problems", "vision changes", "hearing loss", "tinnitus",
        
        # Psychiatric
        "anxiety", "sadness", "irritability", "mood changes", "sleep problems", "insomnia",
        "excessive sleepiness", "nightmares", "panic attacks", "hallucinations",
        
        # Skin
        "rash", "itching", "redness", "dry skin", "skin lesions", "bruising", "pallor",
        "jaundice", "cyanosis", "hair loss", "nail changes",
        
        # Urinary
        "frequent urination", "painful urination", "blood in urine", "decreased urine output",
        "urinary urgency", "incontinence",
    ],
    
    "anatomy": [
        # Cardiovascular
        "heart", "left ventricle", "right ventricle", "left atrium", "right atrium",
        "aorta", "pulmonary artery", "coronary artery", "vena cava", "pulmonary vein",
        
        # Respiratory
        "lungs", "right lung", "left lung", "bronchi", "trachea", "diaphragm", "alveoli",
        "bronchioles", "pleura", "larynx", "pharynx",
        
        # Brain/Nervous
        "brain", "cerebrum", "cerebellum", "brainstem", "spinal cord", "nerves",
        "hippocampus", "amygdala", "thalamus", "hypothalamus", "frontal lobe", "temporal lobe",
        
        # GI
        "esophagus", "stomach", "small intestine", "large intestine", "colon", "rectum",
        "liver", "gallbladder", "pancreas", "spleen",
        
        # Renal
        "kidneys", "right kidney", "left kidney", "ureters", "bladder", "urethra",
        
        # Endocrine
        "thyroid", "parathyroid", "adrenal glands", "pituitary gland", "hypothalamus",
        "pineal gland", "pancreas",
        
        # Musculoskeletal
        "skeletal muscle", "smooth muscle", "cardiac muscle", "bones", "joints", "tendons",
        "ligaments", "spine", "skull", "ribcage", "pelvis",
        
        # Other
        "skin", "eyes", "ears", "nose", "tongue", "teeth", " gums",
        "lymph nodes", "thymus", "bone marrow", "blood vessels", "arteries", "veins",
    ],
    
    "treatment": [
        # Medications
        "pharmacotherapy", "drug therapy", "medication management", "polypharmacy",
        
        # Procedures
        "surgery", "laparoscopic surgery", "endoscopic surgery", "robotic surgery",
        "biopsy", "drainage", "catheterization", "intubation", "ventilation",
        
        # Therapy
        "physical therapy", "occupational therapy", "speech therapy", "respiratory therapy",
        "chemotherapy", "radiation therapy", "immunotherapy", "hormone therapy",
        "gene therapy", "cell therapy", "stem cell therapy",
        
        # Dialysis
        "hemodialysis", "peritoneal dialysis", "kidney transplant",
        
        # Cardiac
        "angioplasty", "stent placement", "pacemaker insertion", "defibrillation",
        "cardioversion", "coronary bypass surgery", "heart transplant",
        
        # Mental Health
        "psychotherapy", "cognitive behavioral therapy", "dialectical behavior therapy",
        "electroconvulsive therapy", "transcranial magnetic stimulation", "group therapy",
        
        # Alternative
        "acupuncture", "massage therapy", "chiropractic", "herbal medicine", "meditation",
        
        # Palliative
        "palliative care", "hospice care", "pain management", "supportive care",
        
        # Lifestyle
        "diet therapy", "exercise therapy", "weight management", "smoking cessation",
        "alcohol moderation", "stress management", "sleep hygiene",
    ],
    
    "test": [
        # Blood Tests
        "complete blood count", "basic metabolic panel", "comprehensive metabolic panel",
        "lipid panel", "liver function tests", "kidney function tests", "thyroid function tests",
        "hba1c", "glucose test", "electrolyte panel", "coagulation panel", "arterial blood gas",
        
        # Imaging
        "x-ray", "ct scan", "mri", "ultrasound", "echocardiogram", "electrocardiogram",
        "pet scan", "bone scan", "mammogram", "colonoscopy", "endoscopy", "bronchoscopy",
        
        # Urine/Stool
        "urinalysis", "urine culture", "stool occult blood", "stool culture",
        
        # Biopsy
        "tissue biopsy", "bone marrow biopsy", "skin biopsy", "liver biopsy", "kidney biopsy",
        
        # Other
        "electroencephalogram", "electromyography", "stress test", "allergy test",
        "biomarker test", "genetic test", " PCR test", "rapid antigen test",
    ],
    
    "pathogen": [
        # Bacteria
        "staphylococcus aureus", "streptococcus pneumoniae", "escherichia coli", "salmonella",
        "klebsiella pneumoniae", "pseudomonas aeruginosa", "neisseria gonorrhoeae", "treponema pallidum",
        "mycobacterium tuberculosis", "clostridium difficile", "helicobacter pylori",
        
        # Virus
        "influenza virus", "hiv", "hepatitis b virus", "hepatitis c virus", "sars-cov-2",
        "respiratory syncytial virus", "measles virus", "mumps virus", "rubella virus",
        "herpes simplex virus", "varicella zoster virus", "ebola virus", "dengue virus",
        
        # Fungi
        "candida albicans", "aspergillus", "cryptococcus neoformans", "pneumocystis jirovecii",
        
        # Parasites
        "plasmodium falciparum", "giardia lamblia", "entamoeba histolytica", "trichinella",
    ]
}

# ============ RELATIONSHIP PATTERNS ============

RELATIONSHIPS = {
    "disease": {
        "has_symptom": ["symptom"],
        "treated_by": ["drug", "treatment"],
        "affects": ["anatomy"],
        "diagnosed_by": ["test"],
        "caused_by": ["pathogen"],
        "isa": ["disease"],
    },
    "drug": {
        "treats": ["disease"],
        "has_side_effect": ["symptom"],
        "interacts_with": ["drug"],
        "isa": ["drug"],
    },
    "treatment": {
        "treats": ["disease"],
        "used_for": ["disease"],
    },
    "pathogen": {
        "causes": ["disease"],
        "affects": ["anatomy"],
    },
    "test": {
        "detects": ["disease"],
        "identifies": ["pathogen"],
    }
}

def generate_comprehensive_kg():
    """Generate comprehensive medical KG"""
    
    terms = []
    term_id_map = {}
    relationships = []
    seen_rels = set()
    
    # Generate terms with IDs
    print("Generating medical terms...")
    
    for category, term_list in MEDICAL_TERMS.items():
        cat_prefix = category[:2].upper()
        for i, term in enumerate(term_list):
            term_id = f"{cat_prefix}{str(i).zfill(5)}"
            terms.append({
                "id": term_id,
                "text": term,
                "category": category
            })
            term_id_map[(term.lower(), category)] = term_id
    
    print(f"Generated {len(terms)} unique terms")
    
    # Count categories
    cat_counts = {}
    for t in terms:
        cat_counts[t['category']] = cat_counts.get(t['category'], 0) + 1
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")
    
    # Generate relationships
    print("\nGenerating relationships...")
    
    # Category-based relationships
    for category, relations in RELATIONSHIPS.items():
        source_terms = [t for t in terms if t['category'] == category]
        
        for source in source_terms:
            for rel_type, target_categories in relations.items():
                for target_cat in target_categories:
                    targets = [t for t in terms if t['category'] == target_cat]
                    
                    # Select 1-3 targets per source
                    num_targets = min(len(targets), random.randint(1, 3))
                    selected_targets = random.sample(targets, num_targets) if targets else []
                    
                    for target in selected_targets:
                        key = (source['id'], rel_type, target['id'])
                        if key not in seen_rels:
                            relationships.append({
                                "head": source['id'],
                                "relation": rel_type,
                                "tail": target['id']
                            })
                            seen_rels.add(key)
    
    # Additional random relationships for variety
    all_pairs = [(t1['id'], t2['id']) for t1 in terms for t2 in terms if t1['id'] != t2['id']]
    random.shuffle(all_pairs)
    
    extra_rel_types = ["related_to", "associated_with", "may_cause", "may_treat"]
    
    for head_id, tail_id in all_pairs[:30000]:
        if (head_id, tail_id) in seen_rels:
            continue
        
        # Skip if already have relationship
        rel = random.choice(extra_rel_types)
        relationships.append({
            "head": head_id,
            "relation": rel,
            "tail": tail_id
        })
        seen_rels.add((head_id, tail_id))
    
    print(f"Generated {len(relationships)} relationships")
    
    # Count relationship types
    rel_counts = {}
    for r in relationships:
        rel_counts[r['relation']] = rel_counts.get(r['relation'], 0) + 1
    
    print("\nRelationship types:")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")
    
    return {
        "terms": terms,
        "relationships": relationships,
        "categories": CATEGORIES,
        "relation_types": list(set(r['relation'] for r in relationships))
    }


def main():
    print("="*60)
    print("ENHANCED MEDICAL KG GENERATOR")
    print("="*60)
    
    data = generate_comprehensive_kg()
    
    output_path = "/home/harshal/medical-embeddings/medical_kg_enhanced.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  Terms: {len(data['terms'])}")
    print(f"  Relationships: {len(data['relationships'])}")
    print(f"  Categories: {len(data['categories'])}")
    print(f"  Relation types: {len(data['relation_types'])}")


if __name__ == "__main__":
    main()
