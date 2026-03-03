import json
import random

random.seed(42)

CATEGORIES = ["disease", "drug", "symptom", "anatomy", "treatment"]
RELATION_TYPES = ["has_symptom", "treated_by", "affects", "isa", "has_location", "interacts_with", "causes", "prevents", "treats"]

DISEASE_PREFIXES = ["acute", "chronic", "severe", "mild", "primary", "secondary", "viral", "bacterial", "autoimmune", "genetic", "idiopathic", "infectious", "inflammatory", "degenerative", "metabolic", "cardiovascular", "respiratory", "gastrointestinal", "neurological", "musculoskeletal"]
DISEASE_SUFFIXES = ["type 1", "type 2", "onset", "recurrent", "refractory", "progressive", "early stage", "advanced", "terminal", "mild", "moderate", "severe"]

BASE_DISEASES = [
    "diabetes", "hypertension", "asthma", "pneumonia", "tuberculosis", "malaria", "dengue", "influenza",
    "cancer", "leukemia", "lymphoma", "melanoma", "carcinoma", "sarcoma",
    "heart disease", "arrhythmia", "heart failure", "cardiomyopathy", "angina", "myocardial infarction",
    "stroke", "hemorrhage", "thrombosis", "embolism", "aneurysm",
    "copd", "bronchitis", "emphysema", "fibrosis", "pulmonary embolism",
    "gastritis", "ulcer", "gastroenteritis", "hepatitis", "cirrhosis", "pancreatitis", "cholecystitis", "appendicitis", "colitis",
    "arthritis", "osteoporosis", "fibromyalgia", "lupus", "scleroderma", "rheumatoid arthritis",
    "meningitis", "encephalitis", "polio", "measles", "mumps", "rubella",
    "hiv", "aids", "herpes", "syphilis", "gonorrhea", "chlamydia",
    "anemia", "hemophilia", "thrombocytopenia", "leukopenia",
    "hypothyroidism", "hyperthyroidism", "addison disease", "cushing syndrome",
    "parkinson", "alzheimer", "dementia", "epilepsy", "multiple sclerosis", "als",
    "depression", "anxiety", "bipolar", "schizophrenia", "ocd", "ptsd", "autism",
    "eczema", "psoriasis", "acne", "urticaria", "cellulitis",
    "cataracts", "glaucoma", "macular degeneration", "conjunctivitis", "retinitis",
    "hearing loss", "otitis", "tinnitus", "vertigo", "sinusitis", "rhinitis",
    "kidney failure", "nephritis", "kidney stones", "cystitis", "urethritis",
    "infertility", "endometriosis", "obesity", "malnutrition",
    "allergies", "anaphylaxis", "hay fever", "insomnia", "sleep apnea",
    "migraine", "headache", "varicose veins", "hemorrhoids"
]

DRUG_PREFIXES = ["oral", "intravenous", "topical", "inhaled", "subcutaneous", "intramuscular", "rectal", "ophthalmic", "nasal", "otic"]
DRUG_FORMS = ["tablet", "capsule", "injection", "solution", "suspension", "cream", "ointment", "gel", "patch", "spray", "drops", "inhaler", "syrup", "powder", "suppository"]

BASE_DRUGS = [
    "insulin", "metformin", "glipizide", "glibenclamide", "sitagliptin", "liraglutide",
    "lisinopril", "enalapril", "ramipril", "valsartan", "losartan", "telmisartan",
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "atenolol", "metoprolol", "propranolol",
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "fenofibrate",
    "warfarin", "heparin", "rivaroxaban", "apixaban", "dabigatran", "clopidogrel", "aspirin",
    "omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "famotidine",
    "metronidazole", "ciprofloxacin", "azithromycin", "amoxicillin", "cephalexin", "doxycycline",
    "ibuprofen", "naproxen", "diclofenac", "meloxicam", "acetaminophen", "tramadol", "morphine",
    "prednisone", "dexamethasone", "hydrocortisone", "methylprednisolone",
    "levothyroxine", "methimazole", "propylthiouracil", "hydrochlorothiazide", "furosemide",
    "spironolactone", "budesonide", "fluticasone", "montelukast", "salbutamol",
    "amitriptyline", "fluoxetine", "sertraline", "escitalopram", "paroxetine", "venlafaxine",
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "zolpidem",
    "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole",
    "levodopa", "carbidopa", "selegiline", "ropinirole", "pramipexole",
    "gabapentin", "pregabalin", "carbamazepine", "valproic acid", "lamotrigine",
    "sumatriptan", "ergotamine", "rizatriptan", "zolmitriptan"
]

SYMPTOM_ADJECTIVES = ["mild", "moderate", "severe", "acute", "chronic", "persistent", "recurrent", "intermittent", "sudden", "gradual", "worsening"]

BASE_SYMPTOMS = [
    "fever", "headache", "fatigue", "weakness", "dizziness", "vertigo", "lightheadedness",
    "chest pain", "palpitations", "shortness of breath", "cough", "wheezing", "sputum",
    "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating",
    "loss of appetite", "weight loss", "weight gain", "night sweats", "swelling",
    "joint pain", "muscle pain", "cramps", "stiffness",
    "rash", "itching", "hives", "redness", "dry skin",
    "sore throat", "runny nose", "congestion", "sneezing", "watering eyes",
    "blurred vision", "double vision", "eye pain", "sensitivity to light",
    "hearing loss", "ear pain", "ringing in ears",
    "difficulty swallowing", "hoarseness", "mouth sores", "dry mouth", "bad breath",
    "urinary frequency", "urinary urgency", "painful urination", "blood in urine",
    "confusion", "memory loss", "difficulty concentrating", "slurred speech",
    "anxiety", "sadness", "irritability", "mood swings", "panic attacks",
    "insomnia", "excessive sleepiness", "nightmares", "snoring",
    "numbness", "tingling", "burning sensation", "tremor",
    "pallor", "jaundice", "cyanosis", "clubbing",
    "hair loss", "brittle nails", "frequent infections", "slow healing", "easy bruising"
]

ANATOMY_MODIFIERS = ["left", "right", "upper", "lower", "anterior", "posterior", "superior", "inferior", "proximal", "distal", "lateral", "medial"]

BASE_ANATOMY = [
    "heart", "lungs", "liver", "kidneys", "pancreas", "spleen", "stomach", "intestines",
    "brain", "spinal cord", "nerves", "muscles", "bones", "joints", "tendons", "ligaments",
    "blood vessels", "arteries", "veins", "capillaries", "aorta",
    "thyroid", "adrenal glands", "pituitary", "hypothalamus", "pineal gland",
    "esophagus", "gallbladder", "bladder", "prostate", "uterus", "ovaries", "testes",
    "skin", "hair", "nails", "eyes", "ears", "nose", "tongue", "teeth",
    "bronchi", "trachea", "diaphragm", "ribcage", "pelvis", "skull", "spine",
    "cerebrum", "cerebellum", "brainstem", "hippocampus", "amygdala", "thalamus",
    "ventricle", "atrium", "septum", "pulmonary artery", "coronary arteries",
    "hepatic vein", "portal vein", "renal artery", "jugular vein",
    "red blood cells", "white blood cells", "platelets", "plasma", "bone marrow",
    "lymph nodes", "thymus", "tonsils", "cervix", "endometrium", "fallopian tubes",
    "skeletal muscle", "smooth muscle", "cardiac muscle", "cartilage"
]

TREATMENT_MODIFIERS = ["conventional", "alternative", "complementary", "experimental", "invasive", "non-invasive", "minimally invasive"]

BASE_TREATMENTS = [
    "surgery", "chemotherapy", "radiation therapy", "immunotherapy", "hormone therapy",
    "physical therapy", "occupational therapy", "speech therapy", "respiratory therapy",
    "dialysis", "transplant", "blood transfusion",
    "oxygen therapy", "ventilator", "cpap", "bipap", "tracheostomy",
    "cardioversion", "defibrillation", "pacemaker", "angioplasty", "stent placement",
    "drainage", "biopsy", "endoscopy", "colonoscopy", "gastroscopy",
    "electroconvulsive therapy", "transcranial magnetic stimulation", "deep brain stimulation",
    "phototherapy", "laser therapy", "cryotherapy", "ablation",
    "acupuncture", "massage therapy", "chiropractic",
    "psychotherapy", "cognitive behavioral therapy",
    "nutritional counseling", "diet therapy", "exercise therapy", "yoga",
    "antibiotic therapy", "antiviral therapy", "antifungal therapy",
    "immunosuppressive therapy", "corticosteroid therapy", "hormone replacement therapy",
    "vaccination", "immunization", "prophylaxis",
    "palliative care", "hospice care", "pain management", "wound care",
    "rehabilitation", "prosthetics", "orthotics", "wheelchair"
]


def generate_variations(base_terms, prefixes, suffixes, count):
    """Generate variations of base terms to reach target count"""
    variations = list(base_terms)
    
    # Add prefix variations
    for term in base_terms:
        for prefix in prefixes:
            if len(variations) >= count:
                break
            variations.append(f"{prefix} {term}")
        if len(variations) >= count:
            break
    
    # Add suffix variations
    for term in base_terms:
        for suffix in suffixes:
            if len(variations) >= count:
                break
            variations.append(f"{term} {suffix}")
        if len(variations) >= count:
            break
    
    return variations[:count]


def generate_terms():
    """Generate 10K medical terms"""
    target_per_category = 2000
    
    diseases = generate_variations(BASE_DISEASES, DISEASE_PREFIXES, DISEASE_SUFFIXES, target_per_category)
    drugs = generate_variations(BASE_DRUGS, DRUG_PREFIXES, DRUG_FORMS, target_per_category)
    symptoms = generate_variations(BASE_SYMPTOMS, SYMPTOM_ADJECTIVES, [], target_per_category)
    anatomy = generate_variations(BASE_ANATOMY, ANATOMY_MODIFIERS, [], target_per_category)
    treatments = generate_variations(BASE_TREATMENTS, TREATMENT_MODIFIERS, [], target_per_category)
    
    all_terms = []
    term_id_map = {}
    
    categories = [("disease", diseases), ("drug", drugs), ("symptom", symptoms), ("anatomy", anatomy), ("treatment", treatments)]
    
    for cat_name, terms in categories:
        for i, text in enumerate(terms):
            term_id = f"{cat_name[0].upper()}{str(i).zfill(5)}"
            all_terms.append({
                "id": term_id,
                "text": text.strip(),
                "category": cat_name
            })
            term_id_map[(text.strip().lower(), cat_name)] = term_id
    
    return all_terms, term_id_map


def generate_relationships(terms):
    """Generate relationships between terms"""
    relationships = []
    seen = set()
    
    # Group terms by category
    terms_by_cat = {}
    for t in terms:
        cat = t["category"]
        if cat not in terms_by_cat:
            terms_by_cat[cat] = []
        terms_by_cat[cat].append(t)
    
    # Generate positive relationships (meaningful)
    # Disease has_symptom Symptom
    for disease in terms_by_cat["disease"]:
        num_symptoms = random.randint(2, 6)
        for symptom in random.sample(terms_by_cat["symptom"], min(num_symptoms, len(terms_by_cat["symptom"]))):
            key = (disease["id"], "has_symptom", symptom["id"])
            if key not in seen:
                relationships.append({"head": disease["id"], "relation": "has_symptom", "tail": symptom["id"]})
                seen.add(key)
    
    # Disease treated_by Drug/Treatment
    for disease in terms_by_cat["disease"]:
        num_treatments = random.randint(2, 5)
        for treatment in random.sample(terms_by_cat["treatment"], min(num_treatments, len(terms_by_cat["treatment"]))):
            key = (disease["id"], "treated_by", treatment["id"])
            if key not in seen:
                relationships.append({"head": disease["id"], "relation": "treated_by", "tail": treatment["id"]})
                seen.add(key)
        
        for drug in random.sample(terms_by_cat["drug"], min(num_treatments, len(terms_by_cat["drug"]))):
            key = (drug["id"], "treats", disease["id"])
            if key not in seen:
                relationships.append({"head": drug["id"], "relation": "treats", "tail": disease["id"]})
                seen.add(key)
    
    # Disease affects Anatomy
    for disease in terms_by_cat["disease"]:
        num_anatomy = random.randint(1, 4)
        for anatomy in random.sample(terms_by_cat["anatomy"], min(num_anatomy, len(terms_by_cat["anatomy"]))):
            key = (disease["id"], "affects", anatomy["id"])
            if key not in seen:
                relationships.append({"head": disease["id"], "relation": "affects", "tail": anatomy["id"]})
                seen.add(key)
    
    # Generate random relationships - MINIMAL FOR FAST TRAINING
    for _ in range(15000):
        cat1 = random.choice(CATEGORIES)
        cat2 = random.choice(CATEGORIES)
        
        t1 = random.choice(terms_by_cat[cat1])
        t2 = random.choice(terms_by_cat[cat2])
        
        if t1["id"] == t2["id"]:
            continue
        
        rel = random.choice(RELATION_TYPES)
        key = (t1["id"], rel, t2["id"])
        
        if key not in seen:
            relationships.append({"head": t1["id"], "relation": rel, "tail": t2["id"]})
            seen.add(key)
    
    return relationships


def main():
    print("Generating 10K medical terms...")
    terms, term_id_map = generate_terms()
    
    print(f"Generated {len(terms)} terms")
    for cat in CATEGORIES:
        count = len([t for t in terms if t["category"] == cat])
        print(f"  - {cat}: {count}")
    
    print("\nGenerating relationships...")
    relationships = generate_relationships(terms)
    
    print(f"Generated {len(relationships)} relationships")
    
    rel_counts = {}
    for r in relationships:
        rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
    for rel, count in sorted(rel_counts.items()):
        print(f"  - {rel}: {count}")
    
    data = {
        "terms": terms,
        "relationships": relationships,
        "categories": CATEGORIES,
        "relation_types": RELATION_TYPES
    }
    
    output_path = "/home/harshal/medical-embeddings/medical_kg.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"Total terms: {len(terms)}")
    print(f"Total relationships: {len(relationships)}")


if __name__ == "__main__":
    main()
