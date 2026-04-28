from transformers import pipeline

# Biomedical NER
ner = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="none"  # IMPORTANT: we will handle merging
)

def reconstruct_entities(entities, text):
    merged = []
    current = None

    for e in entities:
        if current is None:
            current = e
            continue

        if (
            e["entity"] == current["entity"]
            and e["start"] <= current["end"] + 1
        ):
            current["end"] = e["end"]
            current["word"] = text[current["start"]:current["end"]]
        else:
            merged.append(current)
            current = e

    if current:
        merged.append(current)

    return merged
	

LABEL_MAP = {
    "B-Age": "demographics",
    "B-Sex": "demographics",
    "B-Disease_disorder": "history",
    "B-Sign_symptom": "symptoms",
    "B-Diagnostic_procedure": "tests",
    "B-Lab_value": "findings",
    "B-Medication": "medications",
    "B-Biological_structure": "anatomy",
}

def build_fact_layer(entities):
    structured = {
        "demographics": [],
        "history": [],
        "symptoms": [],
        "tests": [],
        "findings": [],
        "medications": [],
        "anatomy": []
    }

    for e in entities:
        label = e["entity"]
        word = e["word"].strip()

        if label in LABEL_MAP:
            structured[LABEL_MAP[label]].append(word)

    # Deduplicate
    for k in structured:
        structured[k] = list(set(structured[k]))

    return structured
	

from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_KEY")

def llm_structuring(text, ner_output):
    prompt = f"""
You are a clinical AI assistant.

Input clinical text:
{text}

NER extracted facts:
{ner_output}

Tasks:
1. Clean and normalize entities (merge fragments like 'cr ea tinine' → 'creatinine')
2. Expand context (e.g., 'kidney' + 'chronic' → 'chronic kidney disease')
3. Return STRICT JSON with:

{{
  "facts": {{
    "demographics": [],
    "history": [],
    "symptoms": [],
    "tests": [],
    "findings": [],
    "medications": []
  }},
  "inference": {{
    "risk_level": "low|medium|high|critical",
    "likely_conditions": [],
    "reasoning": ""
  }}
}}

Rules:
- Facts must be directly supported by text
- Inference must be clearly separated
- No hallucinations
"""

    response = client.chat.completions.create(
        model="gpt-4.1",   # or your preferred model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
	

def medical_pipeline(text):
    # Step 1: NER raw
    raw_entities = ner(text)

    # Step 2: reconstruct spans
    clean_entities = reconstruct_entities(raw_entities, text)

    # Step 3: fact layer
    fact_layer = build_fact_layer(clean_entities)

    # Step 4: LLM refinement
    final_output = llm_structuring(text, fact_layer)

    return final_output
	
text = """
A 62-year-old female with a history of chronic kidney disease and hypertension 
presents with worsening shortness of breath, fatigue, and bilateral leg swelling.
Blood pressure is 170/100 mmHg. Chest X-ray shows pulmonary edema.
Lab tests reveal elevated creatinine and BNP levels.
Patient is currently taking amlodipine and furosemide.
"""

result = medical_pipeline(text)
print(result)