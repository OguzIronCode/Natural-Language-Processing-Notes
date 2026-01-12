import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# The required dataset is being downloaded

nltk.download('conll2000')

# Upload Conll2000 Data set

train_data=conll2000.tagged_sents("train.txt")
test_data= conll2000.tagged_sents("test.txt")

# hmm training

trainer=hmm.HiddenMarkovModelTrainer()
hmm_tager=trainer.train(train_data)

# Test

test_sentence="I am not going to park".split()
tags=hmm_tager.tag(test_sentence)

print(tags)