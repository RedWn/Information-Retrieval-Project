from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk.stem as ns


def stem(words, mode):
    """
    words is a regular string of words and mode is the type of stemmer to use
    the return is a list of stemmed words
    """
    stemmer = (
        ns.PorterStemmer()
    )  # TODO create a system for switching stemmers on the fly
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words


def get_wordnet_pos(tag_parameter):

    tag = tag_parameter[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(words):
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    return lemmatized_words


def removeStopWords(words):
    filtered_text = []
    for word in words:
        if word not in stopwords.words("English"):
            filtered_text.append(word)

    return filtered_text

def get_alternative(word):
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].lemma_names()[0]  # Choose the first synonym
    else:
        return word  # If no synonyms found, keep the original word
