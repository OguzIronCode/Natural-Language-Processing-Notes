import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Read Dataset
df=pd.read_csv("C:\\Users\\oguzz\\Documents\\GitHub\\Natural-Language-Processing-Notes\\TF_IDF_SpamDataSet\\sms_spam.csv",encoding="latin1")
df2=df.head(2)

# tf-idf
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(df2["v2"])


# Word Cluster
feature_names=vectorizer.get_feature_names_out()
tf_idf_Score=X.mean(axis=0).A1 # Mean tf-idf value

df_tfidf=pd.DataFrame({"word":feature_names,"tfidf_Score":tf_idf_Score})

# Sorted
df_tfidf_sorted=df_tfidf.sort_values(by=("tfidf_Score"),ascending=False)

print(df_tfidf)

print(df_tfidf_sorted)



