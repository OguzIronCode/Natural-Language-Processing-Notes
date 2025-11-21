#Bag of Words(BoW)

from sklearn.feature_extraction.text import CountVectorizer

documents=[
    "Kedi evde",
    "Kedi bahçede"
]
vectorizer=CountVectorizer()

#Digital vector from text
x=vectorizer.fit_transform(documents)

#Word Cluster 
#Expected = ["kedi","evde","bahçede"]
print("Kelime kümesi :", vectorizer.get_feature_names_out())

#Vector Representation
print("Vektör temsili :",x.toarray())

"""
    Word Cluster :["bahçede","evde","kedi"]
    Vector Representation :
    [1,1,0]
    [1,0,1]
    
"""