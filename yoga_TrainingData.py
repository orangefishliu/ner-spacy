import json
import spacy
import sys
from spacy.matcher import Matcher

if len(sys.argv) < 2:
    sys.exit('Too few arguments, please speciify the input file')

filename = sys.argv[1]
# Load the reddit comments
with open(filename, 'r', encoding="utf-8") as f:
    yogaJson = json.load(f)

# Extract the texts from json
# Key could be 'selftext', 'body'
texts = list()
for entry in yogaJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab, validate=True)

# Patterns
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
bodyPart_pattern = [
    {'POS': 'NOUN',
     'LEMMA': {'IN': bodyPart}}]

matcher.add("BODY_PART", None, bodyPart_pattern)

TRAINING_DATA = []

# Create a Doc object for each text in yogaJson
for doc in nlp.pipe(texts):
    # Match on the doc and create a list of matched spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    # Get (start character, end character, label) tuples of matches
    entities = [(span.start_char, span.end_char, "BODY_PART") for span in spans]
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {"entities": entities})
    # Append the example to the training data
    TRAINING_DATA.append(training_example)

# Format and save the training data to a json file
json_string = json.dumps(TRAINING_DATA, indent=4)
with open("yoga_trainingData.json", "w") as outfile:
    outfile.write(json_string)