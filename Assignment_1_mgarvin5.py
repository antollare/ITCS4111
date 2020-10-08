import numpy as np
import pandas as pd
import nltk as nl

##Question 1 - Load data into dataframes
dfEnglish = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_Assignment_1/CONcreTEXT_trial_EN.tsv', sep ='\t')
dfItalian = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111_Assignment_1/CONcreText_trial_IT.tsv', sep ='\t')

##Question 2 - strip the TEXT column of punction, make it all lowercase, and obtain a list of words
tokenizer = nl.RegexpTokenizer(r'\w+')
englishText = ''
for i in range(dfEnglish.get('TEXT').size):
    englishText += dfEnglish.get('TEXT').get(i).lower()
englishTokens = tokenizer.tokenize(englishText)
print('English words:')
print(englishTokens)
italianText = ''
for i in range(dfItalian.get('TEXT').size):
    italianText += dfItalian.get('TEXT').get(i).lower()
italianTokens = tokenizer.tokenize(italianText)
print('Italian words:')
print(italianTokens)

##Question 3 - output number of words in each language
englishWordAmount = len(englishTokens)
print('English word amounts')
print(englishWordAmount)
italianWordAmount = len(italianTokens)
print('Italian word amounts')
print(italianWordAmount)

##Question 4 - output number of Unique words in each language
englishFreqDist = nl.FreqDist(englishTokens)
englishUniqueAmount = len(englishFreqDist)
print('English unique word count:')
print(englishUniqueAmount)
italianFreqDist = nl.FreqDist(italianTokens)
italianUniqueAmount = len(italianFreqDist)
print('Italian unique word count:')
print(italianUniqueAmount)

##Question 5 - print info on the most frequent 25 words, and for the most frequent 25 additional words that start with m
##print - Word, number of times it occurs, rank, probability of occurrence, and product of rank and probability
englishMostCommonWords = englishFreqDist.most_common()
englishCommonWordsInfo = []
for i in range(25):
    word = []
    word.append(englishMostCommonWords[i][0])
    word.append(englishMostCommonWords[i][1])
    word.append(i + 1)
    word.append((word[1]/englishWordAmount)*100)
    word.append((word[2]/100)*word[3])
    englishCommonWordsInfo.append(word)
print("English most common word info:")
print(englishCommonWordsInfo)
##seperate M words from all words
englishMWords = []
for i in range(len(englishTokens)):
    if englishTokens[i].startswith('m'):
            englishMWords.append(englishTokens[i])

englishMWordsFreqDist = nl.FreqDist(englishMWords)
englishMostCommonMWords = englishMWordsFreqDist.most_common(25)
englishCommonMWordsInfo = []
for i in range(len(englishMostCommonMWords)):
    word = []
    word.append(englishMostCommonMWords[i][0])
    word.append(englishMostCommonMWords[i][1])
    ##find rank of the m word
    for x in range(len(englishMostCommonWords)):
        if(word[0] == englishMostCommonWords[x][0]):
            word.append(x+1)
    word.append((word[1]/englishWordAmount)*100)
    word.append((word[2]/100)*word[3])
    englishCommonMWordsInfo.append(word)
print("English most common M word info:")
print(englishCommonMWordsInfo)

italianMostCommonWords = italianFreqDist.most_common()
italianCommonWordsInfo = []
for i in range(25):
    word = []
    word.append(italianMostCommonWords[i][0])
    word.append(italianMostCommonWords[i][1])
    word.append(i + 1)
    word.append((word[1]/italianWordAmount)*100)
    word.append((word[2]/100)*word[3])
    italianCommonWordsInfo.append(word)
print("Italian most common word info:")
print(italianCommonWordsInfo)
##seperate M words from all words
italianMWords = []
for i in range(len(italianTokens)):
    if italianTokens[i].startswith('m'):
            italianMWords.append(italianTokens[i])

italianMWordsFreqDist = nl.FreqDist(italianMWords)
italianMostCommonMWords = italianMWordsFreqDist.most_common(25)
italianCommonMWordsInfo = []
for i in range(len(italianMostCommonMWords)):
    word = []
    word.append(italianMostCommonMWords[i][0])
    word.append(italianMostCommonMWords[i][1])
    ##find rank of the m word
    for x in range(len(italianMostCommonWords)):
        if (word[0] == italianMostCommonWords[x][0]):
            word.append(x + 1)
    word.append((word[1]/italianWordAmount)*100)
    word.append((word[2]/100)*word[3])
    italianCommonMWordsInfo.append(word)
print("Italian most common M word info:")
print(italianCommonMWordsInfo)

##Question 6 -
#This data set adequately satisfies Zipf's law since we can see in both languages that the most common token has substantially
#more uses than the second most common word. The least common words only have one or two uses, which further illustrates Zipf's law.
#'the' was the most common english word an it occured twice as often as the fifth most common, and 10 times as much as the most common m-word, 'more'
#'more' was also the 26th most common word in the set. This shows the large discrepency that we would expect to see with Zipf's law.
#The italian set was about the same, with 'di' being the most common and appearing more than twice as often as the fifth most common
#and almost 13 times more than the most common m-word, 'mentre' which was ranked 43rd.