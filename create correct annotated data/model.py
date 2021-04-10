from label_studio.ml import LabelStudioMLBase
import spacy
from spacy.pipeline import EntityRuler

# This is a main declaration of a machine learning model class
class SpacyNER(LabelStudioMLBase):

    def __init__(self, **kwargs):
    
        # call base class constructor
        super(SpacyNER, self).__init__(**kwargs)


        # Collect all keys from config which will be used to extract data 
        # from a task and to form predictions

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']
        self.key_name = schema['inputs'][0]['value'] # key is "reddit"


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
                             'skin', 'chest', 'head', 'forehead', 'neck', 'hip', 'thigh', 'hair', 'tongue', 'tooth',
                             'back',
                             'waist', 'muscle', 'wrinkle'
                         ]}}]
        }
        patterns = [bodyPart_pattern]

        # load the pre-trained spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Create an Entity Ruler and add patterns
        self.ruler = EntityRuler(self.nlp, overwrite_ents=True)
        self.ruler.add_patterns(patterns)

        # Add the Entity Ruler to the nlp pipeline
        self.nlp.add_pipe(self.ruler, after="ner")

        print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}, key_name={key_name}'.format(
            from_name=self.from_name, to_name=self.to_name, labels=str(self.labels), key_name=self.key_name ))

    ###
    # This is where inference happens: 
    # model returns the list of predictions for the input list of tasks
    ###
    def predict(self, tasks, **kwargs):

        # get model predictions
        predictions = []
        for task in tasks:
            results = []
            text = task['data'][self.key_name]
            doc = self.nlp(text)

            # Add a prediction for each ING entity found
            for ent in doc.ents:
                if ent.label_ == 'BODY_PART':
                    # prediction results for the single task doc
                    result = {
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "labels",
                        "value": {
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "text" : ent.text,
                            "labels": [ent.label_]
                        }
                    }
                    results.append(result)
            
            predictions.append({"result": results})
        
        return predictions

    ###
    # This is where training happens: train the model given list of completions, 
    # then return dict with created links and resources
    ###
    def fit(self, completions, workdir=None, **kwargs):
        # Training is not implemented
        return {}
