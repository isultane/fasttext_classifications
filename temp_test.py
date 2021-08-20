import string
import fasttext
import re
import pandas as pd
from yaml import tokens
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Returns a list of common english terms (words)
def initialize_words():
    content = None
    with open('/home/sultan/Downloads/wordlist.txt') as f: # A file containing common english words
        content = f.readlines()
    return [word.rstrip('\n') for word in content]

def parse_sentence(sentence, wordlist):
    new_sentence = "" # output  
    tokens = re.findall(r'\w+\b', sentence)
 #   tokens = word_tokenize(sentence)
    # convert to lower cases
    tokens = [w.lower() for w in tokens]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub(' ', w) for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    for term in tokens:
        new_sentence += parse_terms(term, wordlist)
        new_sentence += " "

    return " ".join(new_sentence.split())

def parse_terms(term, wordlist):
    words = []
    word = find_word(term, wordlist)  
    while word != None and len(word) > 2:
        words.append(word)            
        if len(term) == len(word): # Special case for when eating rest of word
            break
        term = term[len(word):]
        word = find_word(term, wordlist)
    return " ".join(words)

def find_word(token, wordlist):
    i = len(token) + 1
    while i > 1:
        i -= 1
        if token[:i] in wordlist:
            return token[:i]
    return None 

wordlist = initialize_words()
sentence = "Support case insensitive mount pathshttp://www.nabble.com/Non-case-sensitive-nice-URL-tf2643746.htmlI use nice URL to mount some page in the usual way :mountBookmarkablePage('/Organization'  Organization.class);That is great  but it will make an error if a user enter the url by hand and decide to not use case sensitive. I mean  I can mount organization without the capital letter  but it would make the same error if the user use Capital after that.Is there a way of using Nice URL that is not case sensitive?ThanksMarc"
print(parse_sentence(sentence, wordlist))


