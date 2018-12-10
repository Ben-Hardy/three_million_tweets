import spacy

nlp = spacy.load('en_core_web_sm')

text = (u"Hello there. How are you today?")
doc = nlp(text)

for i in doc.ents:
    print(i.text, i.label_)


