from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

# Örnek veri seti

corpus =[
    
    "I love you",
    "I love apple",
    "I love programing",
    "you love me",
    "She loves apple",
    "They love you",
    "I love you and you love me"
]

# Tokenize 

tokens=[word_tokenize(sentence.lower()) for sentence in corpus]
#print(tokens)

# n gram -> n:2
bigrams=[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list,2)))
    
#print(bigrams)

# bigrams frekans counter

bigrams_freq=Counter(bigrams)
#print(bigrams_freq)


# n gram -> n:3
trigrams=[]
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list,3)))
    
#print(trigrams)

trigrams_freq=Counter(trigrams)

#print(trigrams_freq)


#"I love" bigramından sonra "you" veya "apple" gelme olasılıklarını hesapla
bigram=("i","love")
prob_you=trigrams_freq[("i","love","you")]/bigrams_freq[bigram]
prob_apple=trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]

print("You kelimesinin olma olsaılığı :",prob_you)
print("Apple kelimesinin olma olsaılığı :",prob_apple)

