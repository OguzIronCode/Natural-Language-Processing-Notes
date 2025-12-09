'''
Word Embedding 
    1)Word2Vec (Google)
    2)FastText(Facebook)

'''

import pandas as pd
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Example data

sentences=[
    
    "Kedi bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsızdır",
    "Köpekler sadık ve dost canlısıdır",
    "Hayvanlar insanlar için iyi arkadaşlardır"
]

#Tokenized Process

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences ]


#Word2Vec
#sg=0 (BagOfWord),sg=1 (N-Gram) Algorithm

word2vector_Model=Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)


#FastText

fast_text_Model=FastText(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)

def plot_words_embedding(model,title):
    word_vectors=model.wv
    words=list(word_vectors.index_to_key)[:1000]
    vectors=[word_vectors[word] for word in words]
    
    #PCA
    
    pca=PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)

    #3D Visualization
    fig=plt.figure(figsize=(12,8))
    ax=fig.add_subplot(111,projection="3d")
    
    #Draw Vector
    
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])
    
    #Word tag
    
    for i,word in enumerate(words):   
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=12)
    
    ax.set_title(title)  
    
    ax.set_xlabel("Component 1 :") 
    ax.set_ylabel("Component 2 :")  
    ax.set_zlabel("Component 3 :") 
    
    
    plt.show()
    
    
plot_words_embedding(word2vector_Model,"Word2Vec")
plot_words_embedding(fast_text_Model,"FastText")








