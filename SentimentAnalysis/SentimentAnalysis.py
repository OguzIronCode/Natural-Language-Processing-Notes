# import libraries

import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,classification_report


nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


df=pd.read_csv("C:\\Users\\oguzz\\Documents\\GitHub\\Natural-Language-Processing-Notes\\SentimentAnalysis\\amazon.csv")

# text cleaning and preprocessing

def preprocess_text(text):
    
    # tokenize
    tokens=nltk.word_tokenize(text.lower())
    
    # stop words
    filtered_tokens=[token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens=[lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join words
    processed_text=" ".join(lemmatized_tokens)
    
    return preprocess_text

df["reviewText2"]=df["reviewText"].apply(preprocess_text)

 
# nltk sentiment analyzer

analyzer=SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores=analyzer.polarity_scores(text)
    
    sentiment=1 if scores["pos"]>0 else 0
    return sentiment

df["sentiment"]=df["reviewText2"].apply(get_sentiment)
    

# evaluation - test
print(confusion_matrix(df["Positive"],df["sentiment"]))
print(classification_report(df["Positive"],df["sentiment"]))
