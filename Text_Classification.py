##Michael Garvin - 10/8/20
import pandas as pd
import nltk as nl
import sklearn as sk

##Question 1 - add a column to the dataframes of both languages that is the language
dfEnglish = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111/Text_Classification_Assignment/CONcreTEXT_trial_EN.tsv', sep ='\t')
dfItalian = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111/Text_Classification_Assignment/CONcreTEXT_trial_IT.tsv', sep ='\t')

language = []
for i in range(len(dfEnglish.index)):
    language.append("ENGLISH")
dfEnglish['LANGUAGE'] = language

language = []
for i in range(len(dfItalian.index)):
    language.append("ITALIAN")
dfItalian['LANGUAGE'] = language

dfAll = pd.concat([dfEnglish, dfItalian])

##Question 2 - create a training set using "all rows" of the "TEXT" column
#tokenize the text
count_vect = sk.feature_extraction.text.CountVectorizer()
text_train_counts = count_vect.fit_transform(dfAll.get("TEXT"))

#obtain frequencies of the text
tfidf_transformer = sk.feature_extraction.text.TfidfTransformer(use_idf=False).fit(text_train_counts)
text_tran_tfidf = tfidf_transformer.transform(text_train_counts)

##Question 3 - Train and fit a Multinomial Naive Bayes algorithm, target is LANGUAGE
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(text_tran_tfidf, dfAll.get("LANGUAGE"))

##Question 4 - predict the language of two given phrases
test = ['Why does a rose smell sweet?','Pensa ai tuoi sentimenti di amore.']

test_counts = count_vect.transform(test)

test_tfidf = tfidf_transformer.transform(test_counts)

predicted = clf.predict(test_tfidf)

print("Predictions of given text:")
for i in range(len(test)):
    print(f"{test[i]} => {predicted[i]}")

##Question 5 - test model on 5 original sentences from both languages
#English sentences
test = ['How much wood could a woodchuck chuck, if a woodchuck could chuck wood?']
test.append('Mary had a little lamb, whose fleece was white as snow.')
test.append('I went to the store yesterday.')
test.append('Where did I leave my keys?')
test.append('There are two squirrels chasing each other on a tree outside my window.')
#Italian sentences
test.append('Il mercato da solo non risolve tutto, benché a volte vogliano farci credere questo dogma di fede neoliberale.')
test.append("Mi dispiace, ma non parlo bene l'italiano.")
test.append('Mi innervosisco sempre quando parlo in italiano.')
test.append('Mi farebbe un assortimento dei piatti migliori?')
test.append('Come posso arrivarci?')

#format the data and make the prediction
test_counts = count_vect.transform(test)

test_tfidf = tfidf_transformer.transform(test_counts)

predicted = clf.predict(test_tfidf)

print("Predictions of original text:")
for i in range(len(test)):
    print(f"{test[i]} => {predicted[i]}")

##Extra credit - create a sentence that the model gets wrong
test = ["I saw this italian sentence: Non si ferma l'offensiva dell'Azerbaigian ai danni della Repubblica autonoma del Nagorno-Karabakh, nonostante ieri sia partito lo sforzo diplomatico che dovrebbe fermare le ostilità."]

test_counts = count_vect.transform(test)

test_tfidf = tfidf_transformer.transform(test_counts)

predicted = clf.predict(test_tfidf)

print("Prediction of extra credit text:")
for i in range(len(test)):
    print(f"{test[i]} => {predicted[i]}")