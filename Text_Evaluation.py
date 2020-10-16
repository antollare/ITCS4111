##Michael Garvin - 10/15/20
import pandas as pd
import numpy as np
import nltk as nl
import sklearn as sk

##Question 1 - add a column to the dataframes of both languages that is the language
dfEnglish = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111/Text_Classification_Assignment/CONcreTEXT_trial_EN.tsv', sep ='\t')
dfItalian = pd.read_csv('C:/Users/garvi/PycharmProjects/ITCS4111/Text_Classification_Assignment/CONcreTEXT_trial_IT.tsv', sep ='\t')

dfAll = pd.concat([dfEnglish, dfItalian])

#create new CONCRETE column LOW if MEAN <= 4 HIGH if MEAN > 4
dfAll["CONCRETE"] = np.where(dfAll.get("MEAN") <= 4, 'LOW', 'HIGH')

##Question 2 - Train Test Split - 80% train, 20% test
train, test = sk.model_selection.train_test_split(dfAll, test_size = .2)

##Question 3 - Majority Class Baseline
#deterimine majority class
highCount = 0
lowCount = 0
values = np.where(dfAll.get("CONCRETE") == "LOW","low","high")
for i in range(len(values)):
    if(values[i] == 'high'):
        highCount += 1
    else:
        lowCount += 1
if highCount > lowCount:
    majorityClass = 'HIGH'
else:
    majorityClass = 'LOW'

predicted = test.copy()
predicted.CONCRETE = majorityClass
print("Majority Class Baseline metrics: ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get("CONCRETE"),target_names=["HIGH","LOW"], zero_division=0))

##Question 4 - Target Length Baseline if TARGET length >= 5 HIGH
length_checker = np.vectorize(len)
predictedTLB = np.where(length_checker(test.get("TARGET")) >= 5, "HIGH","LOW")
del predicted['CONCRETE']
predicted['CONCRETE'] = predictedTLB
print("Target Length Baseline metrics: ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get('CONCRETE'), target_names=["HIGH","LOW"], zero_division=0))

##Question 5 - Naive Bayes Classifier
count_vect = sk.feature_extraction.text.CountVectorizer()
text_train_counts = count_vect.fit_transform(train.get("TEXT"))

#obtain frequencies of the text
tfidf_transformer = sk.feature_extraction.text.TfidfTransformer(use_idf=False).fit(text_train_counts)
text_tran_tfidf = tfidf_transformer.transform(text_train_counts)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(text_tran_tfidf, train.get("CONCRETE"))

test_counts = count_vect.transform(test.get('TEXT'))

test_tfidf = tfidf_transformer.transform(test_counts)

predictedNBM = clf.predict(test_tfidf)
del predicted['CONCRETE']
predicted['CONCRETE'] = predictedNBM

print("Naive Bayes Model metrics: ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get("CONCRETE"), target_names=["HIGH","LOW"], zero_division=0))

##Question 6 - comparing performance
# The Naive Bayes Model's accuracy is highly variable based on the train and test split but it is usually more accurate than the Majority Class model
# The Target Length Model was closer to Naive as well. The Majority class model has a very high false positive rate compared to the others, and this
# results in the percision being very low, but the false negative rate is 0 so it raises the recall to 1. Of course the avg are lower due to their
# being no LOW classes. From my testing I would choose a Naive Bayes Model for most application, but I would be interested in seeing if I could determine
# the best target length from the data instead of hard coding a value.

##Question 7 - experiment with 3 other values for Target Length
# I am choosing 3, 6, and 8 these are to see if it is more accurate with shorter length, or higher. I also don't exactly know what is meant by a HIGH
# concrete value, so I am not sure which intuitvly makes sense.
length_checker = np.vectorize(len)
predictedTLB = np.where(length_checker(test.get("TARGET")) >= 3, "HIGH","LOW")
del predicted['CONCRETE']
predicted['CONCRETE'] = predictedTLB
print("Target Length Baseline metrics (HIGH >= 3): ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get('CONCRETE'), target_names=["HIGH","LOW"], zero_division=0))

length_checker = np.vectorize(len)
predictedTLB = np.where(length_checker(test.get("TARGET")) >= 6, "HIGH","LOW")
del predicted['CONCRETE']
predicted['CONCRETE'] = predictedTLB
print("Target Length Baseline metrics (HIGH >= 6): ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get('CONCRETE'), target_names=["HIGH","LOW"], zero_division=0))

length_checker = np.vectorize(len)
predictedTLB = np.where(length_checker(test.get("TARGET")) >= 8, "HIGH","LOW")
del predicted['CONCRETE']
predicted['CONCRETE'] = predictedTLB
print("Target Length Baseline metrics (HIGH >= 8): ")
print(sk.metrics.classification_report(test.get("CONCRETE"),predicted.get('CONCRETE'), target_names=["HIGH","LOW"], zero_division=0))
# It seems to be more accurate with a length of 6, but this doesn't seem more accurate than the 5 model