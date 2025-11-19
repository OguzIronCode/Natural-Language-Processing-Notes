import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")

stemmer=PorterStemmer()

# Example, For Stemming

words=["running","runs","ran","runner","better","go","went"]

stems=[stemmer.stem(w) for w in words]

print("Stem result :",stems)

# Example, For Lemma

lemmatizer=WordNetLemmatizer()

lemmas=[lemmatizer.lemmatize(w,pos='v') for w in words]

print("Lemmatizer result :",lemmas)