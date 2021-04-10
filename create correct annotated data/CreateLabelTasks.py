import spacy
from spacy.pipeline import EntityRuler
import json
import sys

####
# Create labeling tasks for Label Studio. Labeling tasks
# include at least one 'BODY_PART' entity found by
# spaCy's EntitRuler pattern matcher.
#
# https://spacy.io/api/entityruler
# https://spacy.io/usage/rule-based-matching#entityruler
####

if len(sys.argv) < 3:
    sys.exit('Too few arguments, please speciify the input reddit data and the file name to store labeling tasks')

inFile = sys.argv[1]
outFile = sys.argv[2]
# Load the reddit comments
with open(inFile, 'r', encoding="utf-8") as f:
  redditJson = json.load(f)

# Extract the texts from json
# Key could be 'selftext', 'body'
texts = list()
for entry in redditJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# List of some common body parts
bodyPart = [
    'face', 'eye', 'nose', 'mouth', 'ear', 'cheek', 'chin', 'nostril',
    'lip', 'arm', 'hand', 'finger', 'palm', 'wrist', 'forearm', 'elbow', 'shoulder', 'thumb',
    'nail', 'knuckle', 'leg', 'knee', 'shin', 'calf', 'ankle', 'heel', 'foot', 'toe', 'heart',
    'lung', 'vein', 'brain', 'throat', 'liver', 'stomach', 'kidney', 'skeleton', 'rib', 'bone',
    'skin', 'chest', 'head', 'forehead', 'neck', 'hip', 'thigh', 'hair', 'tongue', 'tooth', 'back',
    'waist', 'muscle', 'wrinkle'
]

# The pattern includes a 'POS' tag, 'NOUN', and a 'LEMMA', any of the body part in the list 'bodyPart' above.
# An example of pattern-match: ('knee' 'BODY_PART')
bodyPart_match = [
    {'POS': 'NOUN',
     'LEMMA': {'IN': bodyPart}}]

# Entity Patterns
bodyPart_pattern = {
    'label': 'BODY_PART',
    'pattern': [{'POS': 'NOUN',
     'LEMMA': {'IN': [
    'face', 'eye', 'nose', 'mouth', 'ear', 'cheek', 'chin', 'nostril',
    'lip', 'arm', 'hand', 'finger', 'palm', 'wrist', 'forearm', 'elbow', 'shoulder', 'thumb',
    'nail', 'knuckle', 'leg', 'knee', 'shin', 'calf', 'ankle', 'heel', 'foot', 'toe', 'heart',
    'lung', 'vein', 'brain', 'throat', 'liver', 'stomach', 'kidney', 'skeleton', 'rib', 'bone',
    'skin', 'chest', 'head', 'forehead', 'neck', 'hip', 'thigh', 'hair', 'tongue', 'tooth', 'back',
    'waist', 'muscle', 'wrinkle'
]}}]
}

patterns = [bodyPart_pattern]

# Create an Entity Ruler and add patterns
ruler = EntityRuler(nlp, overwrite_ents=True, validate=True)
ruler.add_patterns(patterns)

# Add the Entity Ruler to the nlp pipeline
nlp.add_pipe(ruler)

# Process texts with the Entity Ruler in the pipelne
# Create a labeling task for each doc that has at least
# one entity of type 'ING'

LABELING_DATA = []
for text in texts:
    doc = nlp(text)

    # get list of labels for this doc
    labels = [ent.label_ for ent in doc.ents if ent.label_ == 'BODY_PART']

    # if the doc has at least one 'BODY_PART' entity,
    # add it to the labeling task list
    if labels:
        # Append doc.text to the Label Studio labeling tasks
        task = {}
        task['reddit'] = doc.text
        LABELING_DATA.append(task)

print("{} tasks created from {} docs.".format(len(LABELING_DATA), len(texts)))

with open(outFile, 'w', encoding="utf-8") as f:
    f.write(json.dumps(LABELING_DATA))
