import spacy

def ner_step(text):
    
    '''
    ner_step function takes in a text and returns a string of possible names separated by forward slash (/).
    If no names were detected, it will return None.
    Else it will return a string of names.
    
    '''


    names = ""
    nlp = spacy.load("en_core_web_sm")
    ner_categories = ["PERSON"]
    doc = nlp(text)
    entites = []
    
    
    for ent in doc.ents:
        if ent.label_ in ner_categories:
            entites.append((ent.text, ent.label_))
    if len(entites) == 0:
        return None
    else:
        for entity, category in entites:
            names += entity + " / "
        return names[:-3]
        
        