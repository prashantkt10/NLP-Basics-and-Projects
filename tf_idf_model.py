import nltk
import re
import heapq
import numpy

paragraph="""Disembodiment of the head usually results in permanent death. But this could change in the future as scientists have managed to partially revive pig brains four hours after the animals were beheaded. A study conducted by Yale researchers, published in the journal Nature, shows that the death of brain cells could be halted as the cellular functions of pig brains were restored hours after death. However, the research team has made it clear that none of the brains showed any kind of organized electrical activity that could be linked to consciousness or awareness. Nevertheless, this experiment has revealed a shocking amount of cellular function to challenge the idea that the brain suffers from irreversible damage within minutes of the blood supply being cut off. To carry out this experiment, the team employed a specially designed solution, called BrainEx, with anti-coagulating and oxygen-carrying properties along with pharmacological agents. BrainEx served as synthetic blood that carried oxygen and drugs to slow down or reverse the death of brain cells. This solution was pumped through the head’s circulatory system. To everyone’s surprise, the brain cells started demonstrating renewed circulation, metabolic response, and even spontaneous synaptic response. The pig brains also began showing a normal response by using the same amount of oxygen as a normal brain. All of this carried on until 10 hours after the decapitation. This experiment has proved that the process of cell death is a gradual one and some of those processes can be either delayed, preserved or even reversed. It also reconfirms the notion that consciousness dissipates quickly within minutes of brain death and the only way to stop that from happening is to restore circulation in the brain immediately."""

#ds1 = nltk.sent_tokenize(paragraph)
dataset = nltk.sent_tokenize(paragraph)

for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])
    
#creating the histogram
word2count={}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
freq_words = heapq.nlargest(100, word2count, key=word2count.get)

#IDF Matrix
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = numpy.log((len(dataset)/doc_count)+1)

#TF Metrics
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in dataset:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if w == word:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
    
# TF-IDF Calculation
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)
    
X = numpy.asarray(tfidf_matrix).transpose()