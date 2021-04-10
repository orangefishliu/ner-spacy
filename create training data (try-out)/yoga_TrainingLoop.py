import spacy
import random
import json

from spacy import displacy

with open("yoga_trainingData.json", encoding="utf8") as f:
    TRAINING_DATA = json.loads(f.read())

nlp = spacy.blank("en")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)
ner.add_label("BODY_PART")

# Start the training
nlp.begin_training()

# Loop for 10 iterations
for itn in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, losses=losses, drop=0.2)
    print(losses)

# Test the model with the hard-coded texts
with open("test.json", encoding="utf8") as f:
    TEST_DATA = json.loads(f.read())

docs = []
for test in TEST_DATA:
    doc = nlp(test['body'])
    docs.append(doc)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# Display the testing results with displacy
displacy.serve(docs, style="ent", options={"ents": ["BODY_PART"]})
