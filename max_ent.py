from nltk.classify import MaxentClassifier

# Train data set

train_data= [
    
    ({"love" : True,"amazing" : True},"Positive"),
    ({"hate":True,"terrible" :True},"Negative"),
    ({"happy" : True,"joy" : True},"Positive"),
    ({"sad":True,"depressed":True},"Negative"),
]

# Maximum Entropy Classifier Training

classifier=MaxentClassifier.train(train_data, max_iter=10)

test_sentence="I hate apples,I love you"
features ={word:(word in test_sentence.lower().split()) for word in ["love","amazing","hate","terrible","happy","sad","depressed","joy"]}
label=classifier.classify(features)

print(label)