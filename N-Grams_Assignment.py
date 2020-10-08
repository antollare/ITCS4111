import numpy as np
import pandas as pd
import nltk as nl
import random as rd

##split function to create unigrams
def split1(word):
    return[char for char in word]

##split function to create bigrams
def split2(word):
    bigrams = []
    for i in range(len(word)):
        #place first letter in its own bigram
        if i == 0:
            bigrams.append(word[i])
        else:
            bigram = word[i-1] + word[i]
            bigrams.append(bigram)
    return bigrams

##Question 1 - Create training and test sets in each langauage 80%-20%
dfEnglish = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_N-Grams_Assignment/CONcreTEXT_trial_EN.tsv', sep ='\t')
dfItalian = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_N-Grams_Assignment/CONcreText_trial_IT.tsv', sep ='\t')

##strip the TEXT column of punction, make it all lowercase, and obtain a list of words
# I am removing capitalization in order to simplify the amount of tokens. I possible downside of this is the removal
# of proper nouns that could provide some info on the language. Could be more accurate as capital letters would be
# comparatively rarer

tokenizer = nl.RegexpTokenizer(r'\w+')
englishText = ''
for i in range(dfEnglish.get('TEXT').size):
    englishText += dfEnglish.get('TEXT').get(i).lower()
englishTokens = tokenizer.tokenize(englishText)

italianText = ''
for i in range(dfItalian.get('TEXT').size):
    italianText += dfItalian.get('TEXT').get(i).lower()
italianTokens = tokenizer.tokenize(italianText)

englishTrainingSet = []
englishTestSet = []
rd.seed()

rd.shuffle(englishTokens)
englishTrainingSetLength = len(englishTokens) * .8
for i in range(len(englishTokens)):
    if(i < englishTrainingSetLength):
        englishTrainingSet.append(englishTokens[i])
    else:
        englishTestSet.append(englishTokens[i])

italianTrainingSet = []
italianTestSet = []

rd.shuffle(italianTokens)
italianTrainingSetLength = len(italianTokens) * .8
for i in range(len(italianTokens)):
    if (i < italianTrainingSetLength):
        italianTrainingSet.append(italianTokens[i])
    else:
        italianTestSet.append(italianTokens[i])

print("English Training set length: ")
print(len(englishTrainingSet))

print("English Test set length: ")
print(len(englishTestSet))

print("Italian Training set length: ")
print(len(italianTrainingSet))

print("Italian Test set length: ")
print(len(italianTestSet))

##Question 2 - Build unigram model for each language
# split words into unigrams
englishUnigramModel = []
for i in range(len(englishTrainingSet)):
    letters = split1(englishTrainingSet[i])
    for char in letters:
        englishUnigramModel.append(char)

italianUnigramModel = []
for i in range(len(italianTrainingSet)):
    letters = split1(italianTrainingSet[i])
    for char in letters:
        italianUnigramModel.append(char)

#obtain freqDist for each language unigrams
englishUnigramFreqDist = nl.FreqDist(englishUnigramModel)
italianUnigramFreqDist = nl.FreqDist(italianUnigramModel)

#test Test sets to determine accuracy
englishResults = []
for word in englishTestSet:
    #split word into unigrams
    unigrams = split1(word)
    #determine probability of each unigram
    englishProbabilities = []
    italianProbabilities = []
    for char in unigrams:
        englishFrequency = englishUnigramFreqDist.get(char)
        italianFrequency = italianUnigramFreqDist.get(char)
        # FreqDist does not return anything if char is not present
        if(type(englishFrequency) is int):
            englishProbabilities.append(float(englishFrequency)/float(len(englishUnigramModel)))
        else:
            englishProbabilities.append(0)
        if (type(italianFrequency) is int):
            italianProbabilities.append(float(italianFrequency)/float(len(italianUnigramModel)))
        else:
            italianProbabilities.append(0)
    #multiply probabilities to determine combined probablity
    englishProbability = 1.0;
    italianProbability = 1.0;
    for i in range(len(englishProbabilities)):
        englishProbability *= englishProbabilities[i]
        italianProbability *= italianProbabilities[i]
    #assign each word to a language word is english if the probability is greater than or equal to the italian
    # using equal to since there are more english writings I would encounter, I will use the same metric for italian
    # results
    if(englishProbability >= italianProbability):
        englishResults.append(1)
    else:
        englishResults.append(0)
# determine accuracy of english test
sum = 0
for x in englishResults:
    sum += x

englishAccuracy = float(sum)/float(len(englishResults))*100

print("Accuracy of English Unigram Test: ")
print(englishAccuracy)

#repeat test for italianTestSet
italianResults = []
for word in italianTestSet:
    #split word into unigrams
    unigrams = split1(word)
    #determine probability of each unigram
    englishProbabilities = []
    italianProbabilities = []
    for char in unigrams:
        englishFrequency = englishUnigramFreqDist.get(char)
        italianFrequency = italianUnigramFreqDist.get(char)
        # FreqDist does not return anything if char is not present
        if(type(englishFrequency) is int):
            englishProbabilities.append(float(englishFrequency)/float(len(englishUnigramModel)))
        else:
            englishProbabilities.append(0)
        if (type(italianFrequency) is int):
            italianProbabilities.append(float(italianFrequency)/float(len(italianUnigramModel)))
        else:
            italianProbabilities.append(0)
    #multiply probabilities to determine combined probablity
    englishProbability = 1.0;
    italianProbability = 1.0;
    for i in range(len(englishProbabilities)):
        englishProbability *= englishProbabilities[i]
        italianProbability *= italianProbabilities[i]
    #assign each word to a language word is english if the probability is greater than or equal to the italian
    # using equal to since there are more english writings I would encounter, I will use the same metric for italian
    # results
    if(englishProbability >= italianProbability):
        italianResults.append(0)
    else:
        italianResults.append(1)
# determine accuracy of english test
sum = 0
for x in italianResults:
    sum += x

italianAccuracy = float(sum)/float(len(italianResults))*100

print("Accuracy of Italian Unigram Test: ")
print(italianAccuracy)

## the unigram model seems to be far more accurate for the italian model than the english model

##Question 3 - do the same thing but with a bigram model
englishBigramModel = []
for i in range(len(englishTrainingSet)):
    bigrams = split2(englishTrainingSet[i])
    for bigram in bigrams:
        englishBigramModel.append(bigram)

italianBigramModel = []
for i in range(len(italianTrainingSet)):
    bigrams = split2(italianTrainingSet[i])
    for bigram in bigrams:
        italianBigramModel.append(bigram)

#obtain freqDist for each language unigrams
englishBigramFreqDist = nl.FreqDist(englishBigramModel)
italianBigramFreqDist = nl.FreqDist(italianBigramModel)

#test Test sets to determine accuracy
englishResults = []
for word in englishTestSet:
    #split word into bigrams
    bigrams = split2(word)
    #determine probability of each bigram
    englishProbabilities = []
    italianProbabilities = []
    for bigram in bigrams:
        englishFrequency = englishBigramFreqDist.get(bigram)
        italianFrequency = italianBigramFreqDist.get(bigram)
        # FreqDist does not return anything if char is not present
        if(type(englishFrequency) is int):
            englishProbabilities.append(float(englishFrequency)/float(len(englishBigramModel)))
        else:
            englishProbabilities.append(0)
        if (type(italianFrequency) is int):
            italianProbabilities.append(float(italianFrequency)/float(len(italianBigramModel)))
        else:
            italianProbabilities.append(0)
    #multiply probabilities to determine combined probablity
    englishProbability = 1.0;
    italianProbability = 1.0;
    for i in range(len(englishProbabilities)):
        englishProbability *= englishProbabilities[i]
        italianProbability *= italianProbabilities[i]
    #assign each word to a language word is english if the probability is greater than or equal to the italian
    # using equal to since there are more english writings I would encounter, I will use the same metric for italian
    # results
    if(englishProbability >= italianProbability):
        englishResults.append(1)
    else:
        englishResults.append(0)
# determine accuracy of english test
sum = 0
for x in englishResults:
    sum += x

englishAccuracy = float(sum)/float(len(englishResults))*100

print("Accuracy of English Bigram Test: ")
print(englishAccuracy)

#repeat test for italianTestSet
italianResults = []
for word in italianTestSet:
    #split word into unigrams
    bigrams = split2(word)
    #determine probability of each unigram
    englishProbabilities = []
    italianProbabilities = []
    for bigram in bigrams:
        englishFrequency = englishBigramFreqDist.get(bigram)
        italianFrequency = italianBigramFreqDist.get(bigram)
        # FreqDist does not return anything if char is not present
        if(type(englishFrequency) is int):
            englishProbabilities.append(float(englishFrequency)/float(len(englishBigramModel)))
        else:
            englishProbabilities.append(0)
        if (type(italianFrequency) is int):
            italianProbabilities.append(float(italianFrequency)/float(len(italianBigramModel)))
        else:
            italianProbabilities.append(0)
    #multiply probabilities to determine combined probablity
    englishProbability = 1.0;
    italianProbability = 1.0;
    for i in range(len(englishProbabilities)):
        englishProbability *= englishProbabilities[i]
        italianProbability *= italianProbabilities[i]
    #assign each word to a language word is english if the probability is greater than or equal to the italian
    # using equal to since there are more english writings I would encounter, I will use the same metric for italian
    # results
    if(englishProbability >= italianProbability):
        italianResults.append(0)
    else:
        italianResults.append(1)
# determine accuracy of english test
sum = 0
for x in italianResults:
    sum += x

italianAccuracy = float(sum)/float(len(italianResults))*100

print("Accuracy of Italian Bigram Test: ")
print(italianAccuracy)

## The bigram model was more accurate than the unigram model for english, but the accuracies remained about even for
# italian. I think that this is due to the already high accuracy of the unigram model for italian. It seems that a
# bigram model is much better for distinguishing english. I think that this is due to the lack of accents in english.
# all of the characters used in english are shared across several languages and there are only a few unique or rare
# letters such as w which does not appear in the italian text. The bigram model was about accurate by about 20 points for english

##Observations - it seems like the unigram model works fine for distinguishing latin. I wonder how it would be if the
# languages were two romances languages like french and latin. Or if they were japanese and chinese which share many
# characters and symbols but different pronounciation. An N-Gram of a couple letters seems similar to phonomes, which
# is how linguist differentiate languages. I wonder if it would be possible to seperate the text by phonomes, but that
# seems complex and hard to automate.