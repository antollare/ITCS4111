import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import re
import matplotlib

def clean(word):
    word = word.strip()
    word = word.lower()
    word = re.sub('[^A-Za-z0-9]+', '', word)
    return word

##Question 1 (Creating and Visualizing Vectors)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#%matplotlib inline

line_count = 0
sentences = []
with open('C:/Users/garvi/PycharmProjects/ITCS4111/Text_Classification_Assignment/CONcreTEXT_trial_EN.tsv','r') as inpFile:
    x = inpFile.readlines()
    for line in x:
        if line is not None or line != '\n':
            words = line.split()
            words = map(lambda x: clean(x), words)
            words = list(filter(lambda x:True if len(x) > 0 else False, words))
            sentences.append(words)

model = Word2Vec(sentences, window=5, size=500, workers=4, min_count=5)


labels = []
tokens = []
for word in model.wv.vocab:
    tokens.append(model[word])
    labels.append(word)

tsne_model = TSNE(perplexity=250, n_components=2, init='pca', n_iter=250)
new_values = tsne_model.fit_transform(tokens)
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))
for i in range(len(x)):
  plt.scatter(x[i],y[i])
  plt.annotate(labels[i],
  xy=(x[i], y[i]),
  xytext=(5, 2),
  textcoords='offset points',
  ha='right',
  va='bottom')
plt.show()




##Question 2 (Model Diagnosis)
#The embeddings do not seem very accurate to me, I had trouble with tokenizing the data and the graph seems to look fine. None of the words that should be in my vocabulary are there though
#I think that this might be an issue with me not correctly using the text column but I am unsure. I would say that from the graph it seems like the right words are similar
#but they might be too similar to each other.


