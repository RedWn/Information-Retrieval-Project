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


def process_capital_punctuation(words):
    new_tokens = []
    for word in words:
        word = word.lower()
        new_tokens.append(word.translate(str.maketrans("", "", string.punctuation)))
    return new_tokens


# # Cache the synsets to avoid redundant processing
# @lru_cache(maxsize=None)
# def get_synsets(word):
#     return wordnet.synsets(word)


# def get_unified_synonym(word):
#     # If the word is a digit, return it as is
#     if word.isdigit():
#         return str(word)

#     # If the word is a numeric word, return it as a number
#     try:
#         return str(w2n.word_to_num(word))
#     except ValueError:
#         # Get the synsets once and reuse
#         synsets = get_synsets(word)
#         if synsets:
#             # Directly access lemma names and count occurrences
#             lemma_names = [
#                 lemma.name() for synset in synsets for lemma in synset.lemmas()
#             ]
#             # Get the most common synonym for the word
#             unified_synonym = max(set(lemma_names), key=lemma_names.count)
#             try:
#                 return str(w2n.word_to_num(unified_synonym))
#             except ValueError:
#                 return unified_synonym.lower()
#     return word.lower()


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
    
# def is_country_name(name):
#     # with open(os.devnull, 'w') as devnull:
#     #     with contextlib.redirect_stderr(devnull):
#             # Try to convert the name to ISO 3166-1 alpha-3 country codes
#     code = coco.convert(names=[name], to='ISO3')
#     # If the conversion result is 'not found', the name is not a country name
#     return code != 'not found'
    
def standardize_country_names_OLD(name):
    converted = coco.convert(names=[name], to='name_short')
    if converted != 'not found':
        return converted
    else:
        return name()
    

def standardize_country_names(name):
    # If the name is in the dictionary, return the standardized form
    if name in country_dict:
        return country_dict[name]
    return name


# def is_roman_numeral(s):
#     # This pattern matches Roman numerals from 1 to 3999
#     pattern = 'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
#     return bool(re.fullmatch(pattern, s, re.IGNORECASE))

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
    # 'scotland': 'united kingdom',
    # 'wales': 'united kingdom',
    'prc': "china",
    'uae': 'united arab emirates',
    'u.a.e.': 'united arab emirates',
    'emirates': 'united arab emirates',
    # Add more mappings as needed...
}
