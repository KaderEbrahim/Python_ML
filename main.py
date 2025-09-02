

from transformers import pipeline
import spacy

hf_ner = pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")

spacy_nlp = spacy.load("en_core_web_sm")

def hybrid_ner(text: str):
    hf_results = hf_ner(text)
    entities = [
        {
            "text": ent["word"],
            "label": ent["entity_group"],
            "score": round(float(ent["score"]), 4),
            "start": ent["start"],
            "end": ent["end"]
        }
        for ent in hf_results
    ]


    has_date = any(ent["label"] == "DATE" for ent in entities)

    if not has_date:

        doc = spacy_nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "score": 1.0,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

    return entities


if __name__ == "__main__":
    sample_text =" Microsoft opened a store in Chennai in 2024. "
    results = hybrid_ner(sample_text)

    for r in results:
        print(r)
