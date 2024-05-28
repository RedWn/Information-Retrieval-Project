from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk.stem as ns
from word2number import w2n
from functools import lru_cache
import string
import country_converter as coco
import roman
import re
import spacy
from number_parser import parse_ordinal


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

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

def remove_single_letters(words):
    processed_words = []
    for word in words:
        # If the word is not a single letter, add it to the processed_words list
        if len(word) > 1 or word.isdigit():
            processed_words.append(word)
    return processed_words


def process_capital_punctuation(words):
    new_tokens = []
    for word in words:
        word = word.lower()
        new_tokens.append(word.translate(str.maketrans("", "", string.punctuation)))
    return new_tokens


# Create a dictionary to store precomputed results
memo = {}
def get_unified_synonym_2(word):
    # Check if the result is already in the dictionary
    if word in memo:
        return memo[word]

    # Otherwise, compute the result as before
    if word.isdigit():
        result = str(word)
    
    # Convert Roman numeral to integer
    elif is_roman_numeral(word):
        result = str(roman.fromRoman(word.upper()))
    
    # Convert ordianl words like "first" to "1st"
    elif is_ordinal(word):
        result = ordinal_word_to_ordinal_number(word.lower())

    else:
    # If the word is a numeric word, return it as a number
        try:
            result = str(w2n.word_to_num(word))
        except ValueError:    
            result = word.lower()
    
    result = standardize_country_names(result)
    # Store the result in the dictionary
    memo[word] = result
    return result



# This pattern matches Roman numerals from 1 to 49
pattern = '(I|II|III|IV|V|VI|VII|VIII|IX|X|XX|XXX|XL|L)?'
def is_roman_numeral(s):
    return bool(re.fullmatch(pattern, s, re.IGNORECASE))


ordinal_words = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth', 'eightieth', 'ninetieth', 'hundredth', 'thousandth']
def is_ordinal(word):
    # Check if the word ends with an ordinal suffix
    if re.fullmatch(r'.*(st|nd|rd|th)$', word, re.IGNORECASE):
        # If it does, check if it's an ordinal word
        if(word in ordinal_words):
            return True
    return False


def ordinal_word_to_ordinal_number(word):
    number = parse_ordinal(word)
    suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][(number % 10 if number % 100 not in [11, 12, 13] else 0)]
    return str(number) + suffix
 

def standardize_country_names(name):
    # If the name is in the dictionary, return the standardized form
    if name in country_dict:
        return country_dict[name]
    return name


country_dict = {
    'usa': 'united states',
    # 'us': 'united states',
    'u.s.': 'united states',
    'u.s.a.': 'united states',
    'america': 'united states',
    'uk': 'united kingdom',
    'u.k.': 'united kingdom',
    'britain': 'united kingdom',
    'england': 'united kingdom',
    'prc': "china",
    'uae': 'united arab emirates',
    'u.a.e.': 'united arab emirates',
    'emirates': 'united arab emirates',
    # Add more mappings as needed...
}
