import spacy 

nlp=spacy.load("en_core_web_sm")

# İncelenecek Kelime

word="This text is an example text. "

# Kelimeyi nlp isleminden geçir

doc=nlp(word)

for token in doc:
    print("Text :" ,token.text)
    print("Lemma :",token.lemma_)
    print("POS :",token.pos_)
    print("Dependency",token.dep_)
    print("Shape :",token.shape_)
    print("Is alpha :",token.is_alpha)
    print("Is stop :",token.is_stop)
    print("Morphology :",token.morph)
    print(f"Is plural : ''{'Number = Plur in token.morph'}")
    print(" ")

