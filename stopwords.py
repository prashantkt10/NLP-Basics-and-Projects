import nltk
nltk.download("stopwords")

MartinSpeech = """I have a dream that one day down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification – one day right there in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers.

I have a dream today.

I have a dream that one day every valley shall be exalted and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together.
"""

sentences = nltk.sent_tokenize(MartinSpeech)
# print(sentences[2])

# Stopwords
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    stemwords = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    sentences[i] = ' '.join(stemwords)

