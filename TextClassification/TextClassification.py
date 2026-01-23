import pandas as pd
import nltk 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Veri Seti iceriye aktarılacak

dt=pd.read_csv("sms_spam.csv", encoding="latin-1")
dt.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
dt.columns=['label','text']
print(dt.isna().sum())

# EDA (Exploratory Data Analysis) ile Missing Value Kontrolu

'''
Text Preprocessing :

    -Remove Special Character
    -Lowercase
    -Token 
    -Remove Stopwords
    -Lemmatize
    


'''

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


text=list(dt["text"])

lemmatizer=WordNetLemmatizer()
corpus=[]

for i in range(len(text)):
    r=re.sub(r"^a-zA-Z","",text[i])
    r=r.lower()
    r=r.split()
    r=[word for word in r if word not in stopwords.words("english")]
    r=[lemmatizer.lemmatize(word) for word in r]
    r=" ".join(r)
    corpus.append(r)
    
dt["text2"]=corpus
    
# Train test split (%67 Train %33 Test Data Set)

X=dt["text2"]
y=dt["label"]

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

# Future Extraction (Bag of Words)

cv=CountVectorizer()
X_train_cv=cv.fit_transform(X_train)

# Classifier Training (Model Training and Evaluation)

dect=DecisionTreeClassifier()
dect.fit(X_train_cv,y_train)

x_test_cv=cv.transform(x_test)

# Prediction

predictions=dect.predict(x_test_cv)
c_matrix= confusion_matrix(y_test,predictions)

# Accuracy hesaplama

toplam=c_matrix.sum()
accuarcy = 100 * (toplam - c_matrix[0,1] - c_matrix[1,0]) / toplam

print("Accuarcy : ", accuarcy)