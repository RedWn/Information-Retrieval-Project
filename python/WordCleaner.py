from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk.stem as ns


def stem(words, mode):
    stemmer = ns.PorterStemmer()
    if mode == "Porter":
        stemmer = ns.PorterStemmer()
    elif mode == "Snowball":
        stemmer = ns.SnowballStemmer(language="english")
    elif mode == "Lancaster":
        stemmer = ns.LancasterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def get_wordnet_pos(tag_parameter):
    tag = tag_parameter[0].upper()
    tag_dict = {
        # redwan should we expand this list to take more options TODO
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    # if the input tag is not recognized it defaults to treating it as a noun.
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(words):
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    return lemmatized_words


def remove_stop_words(words):
    filtered_text = []
    for word in words:
        if word not in stopwords.words("English"):
            filtered_text.append(word)

    return filtered_text


def get_alternative(word):
    treebank_pos = pos_tag([word])[0][1]  # Get the POS tag
    synsets = wordnet.synsets(word, pos=get_wordnet_pos(treebank_pos))
    # synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].lemma_names()[0].lower()  # Choose the first synonym
    else:
        return word  # If no synonyms found, keep the original word


def get_alternative_hypernym(word):
    synsets = wordnet.synsets(word)
    if synsets:
        # Get the first synset (sense)
        first_synset = synsets[0]
        # Get the hypernyms for the first synset
        hypernyms = first_synset.hypernyms()
        if hypernyms:
            # Choose the first hypernym
            return hypernyms[0].lemma_names()[0]
        else:
            # If no hypernyms found, return the original word
            return word
    else:
        # If no synsets found, keep the original word
        return word


def get_unified_synonym(word):
    synonyms = set()
    treebank_pos = pos_tag([word])[0][1]  # Get the POS tag
    for synset in wordnet.synsets(word, pos=get_wordnet_pos(treebank_pos)):
        synonyms.update(lemma.name() for lemma in synset.lemmas())

    # Choose a unified form (e.g., the most common synonym)
    if synonyms:
        word_counts = {
            syn: sum(syn in synset.lemma_names() for synset in wordnet.synsets(word))
            for syn in synonyms
        }
        unified_synonym = max(word_counts, key=word_counts.get)
    else:
        unified_synonym = word  # If no synonyms found, keep the original word

    return unified_synonym.lower()


def synonym_map_corpus(corpus):
    # Update the dataset with synonyms words
    mapped_dataset = {}
    for key, words in corpus.items():
        mapped_dataset[key] = [get_unified_synonym(word) for word in words]
    return mapped_dataset