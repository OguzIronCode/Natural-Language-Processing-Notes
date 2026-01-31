import nltk
from nltk.wsd import lesk

# Download the necessary nltk packages
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

# Firt Sentence

sentence1="I wen to the bank to deposit money"
word1="bank"
sense1=lesk(nltk.word_tokenize(sentence1),word1)

print("Sentence",sentence1)
print("Word",word1)
print("Sense",sense1.definition())

sentence2="The river bank was flooded after the heavy rain"
word2="bank"
sense2=lesk(nltk.word_tokenize(sentence2),word2)

print("Sentence",sentence1)
print("Word",word1)
print("Sense",sense1.definition())

'''
It is not sufficient in terms of meaning

'''