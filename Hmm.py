import nltk
from  nltk.tag import hmm


# Example Data Set

train_data=[
    
    [("I","PRON"),("am","VERB"),("a","DT"),("student","NOUN")],
    [("You","PRON"),("are","VERB"),("a","DT"),("teacher","NOUN")]]


# hmm training

trainer=hmm.HiddenMarkovModelTrainer()
hmm_tragger=trainer.train(train_data)

# New Sentence

test_sentence ="I am a teacher".split()
tags =hmm_tragger.tag(test_sentence)

print("Etiketli cümle :",tags)
