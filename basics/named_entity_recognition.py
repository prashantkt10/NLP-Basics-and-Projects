import nltk

paragraph = "I come from India which is a very great country"

words = nltk.word_tokenize(paragraph)

tagged_words = nltk.pos_tag(words)

namedEnt = nltk.ne_chunk(tagged_words)

namedEnt.draw()