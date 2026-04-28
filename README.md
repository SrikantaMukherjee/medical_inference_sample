# medical_inference_sample

This simple python program reads a body of medical information stream about a patient and maps it into a segmented factual data and intelligent inference.
It uses techniques in NER (named entity recognition) using classical medical BERT model. This is followed by general cleanup of the entities that are recognized in the text.

Subsequently, the program calls a LLM model to improve it as a medical assistant to a doctor/patient and process the information into a structured dataset of factual and inference information.

Facts{patient_profile, symptoms, findings} 
Inference{risk, likely_condition, explanations} using GPT-4.x
