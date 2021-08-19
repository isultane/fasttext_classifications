import string
import fasttext
import re
import pandas as pd
from utilities import get_keywords
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
    tokens = word_tokenize(sentence)
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
    while word != None and len(term) > 0:
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
sentence = "NullPointerException after deserialize wicket.util.concurrent.CopyOnWriteArrayListwicket.feedback.FeedbackMessages by default using wicket.util.concurrent.CopyOnWriteArrayList for storage. however CopyOnWriteArrayList internally use a transient Object[] array_ without checking null and lazy initialization. This may cause NullPointerException after session replication or the like. Below is stack trace while testing terracotta session clustering :WicketMessage: unable to get object  model: Model:classname=&#91;wicket.feedback.FeedbackMessagesModel&#93;:attached=true  called with component [MarkupContainer &#91;Component id = messages  page = ngc.wicket.pages.MainPage   path = 7:globalFeedback:feedbackul:messages.FeedbackPanel$MessageListView  isVisible = true  isVersioned = false&#93;]Root cause:java.lang.NullPointerExceptionat wicket.util.concurrent.CopyOnWriteArrayList.size (CopyOnWriteArrayList.java:152)at wicket.feedback.FeedbackMessages.messages(FeedbackMessages.java:258)at wicket.feedback.FeedbackMessagesModel.onGetObject(FeedbackMessagesModel.java:101)at wicket.model.AbstractDetachableModel.getObject (AbstractDetachableModel.java:104)at wicket.Component.getModelObject(Component.java:990)at wicket.markup.html.panel.FeedbackPanel.updateFeedback(FeedbackPanel.java:234)at wicket.Page$2.component (Page.java:372)at wicket.MarkupContainer.visitChildren(MarkupContainer.java:744)at wicket.Page.renderPage(Page.java:368)"
print(parse_sentence(sentence, wordlist))


