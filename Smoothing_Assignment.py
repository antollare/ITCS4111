import pandas as pd
import nltk as nl
import random as rd
import string

def generateSentence(model,character):
    text = ''
    for i in range(100):
        text += character
        character = model.generate(1,text)
        if character == '</s>':
            break
    return text

##Question 1- create english language models, one with laplace smoothing and one with linear interpolation
dfEnglish = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_N-Grams_Assignment/CONcreTEXT_trial_EN.tsv', sep ='\t')
dfItalian = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_N-Grams_Assignment/CONcreText_trial_IT.tsv', sep ='\t')

#create the Laplace model
train, vocab = nl.lm.preprocessing.padded_everygram_pipeline(2,dfEnglish.get('TEXT'))
laplaceModel = nl.lm.models.Laplace(2)
laplaceModel.fit(train, vocab)

#create KneserNeyInterpolated model
train, vocab = nl.lm.preprocessing.padded_everygram_pipeline(2,dfEnglish.get('TEXT'))
interpolatedModel = nl.lm.KneserNeyInterpolated(2)
interpolatedModel.fit(train, vocab)

##Question 2- generate 100 character sentence based off of input
letters = string.ascii_lowercase
print("LaPlace Model Generation, bigram:")
for i in range(5):
    seed = rd.choice(letters)
    print(generateSentence(laplaceModel,seed))

print("Interpolated Model Generation, bigram:")
for i in range(5):
    seed = rd.choice(letters)
    print(generateSentence(interpolatedModel,seed))

##Question 3- do the same except with trigrams
train, vocab = nl.lm.preprocessing.padded_everygram_pipeline(3,dfEnglish.get('TEXT'))
laplaceModel = nl.lm.models.Laplace(3)
laplaceModel.fit(train, vocab)

#create KneserNeyInterpolated model
train, vocab = nl.lm.preprocessing.padded_everygram_pipeline(3,dfEnglish.get('TEXT'))
interpolatedModel = nl.lm.KneserNeyInterpolated(3)
interpolatedModel.fit(train, vocab)

##Question 2- generate 100 character sentence based off of input
letters = string.ascii_lowercase
print("LaPlace Model Generation, trigram:")
for i in range(5):
    seed = rd.choice(letters)
    print(generateSentence(laplaceModel,seed))

print("Interpolated Model Generation, trigram:")
for i in range(5):
    seed = rd.choice(letters)
    print(generateSentence(interpolatedModel,seed))