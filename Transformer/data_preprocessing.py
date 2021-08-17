import collections
from collections import Counter
import numpy as np, pandas as pd
import itertools


import re

## uncomment nltk for now - see if this affects the result
#import nltk
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()

###########################################################################################
############# PREPROCESSING TO LOG LINES
###########################################################################################

def preprocess_to_log_lines(log, keep_numbers):
    ### Split lines
    log=log.splitlines()
    
    ### Lemmatize
    #log=[lemmatizer.lemmatize(log[i]) for i in range(len(log))]
    
    ### remove the "weird names"
    log=[re.sub('\ufeff','',log[i]) for i in range(len(log))]
    log=[' '.join(log[i].split()) for i in range(len(log))]
    log=[re.sub(' +', ' ', log[i]) for i in range(len(log))]
    logical=[re.search('^\<|^[0-9]|^\#', log[i]) for i in range(len(log))]

    for i in range(len(log)):
        if logical[i] is None:
            log[i]=''
    
    #bool_list=[x in weirdNames for x in log]
    #res=list(compress(range(len(bool_list)), bool_list))
    #res

    #for i in res:
    #    log[i]=''
        
    ### to lower
    log=[log[i].lower() for i in range(len(log))]
    ###
    
    ### remove numbers if keep_numbers is False
    if not keep_numbers:
        log=[re.sub('[0-9]', '', log[i]) for i in range(len(log))]
    ###
    
    ### remove \r\n
    log=[re.sub('\\r\\n', '', log[i]) for i in range(len(log))]
    ###
    
    ### remove all non-alphanumeric characters
    log=[re.sub('[\W_]+', ' ',log[i]) for i in range(len(log))]
    
    ### remove all non-english characters
    temp=[re.search('[\u0080-\uFFFF]+', log[i]) for i in range(len(log))]
    for i in range(len(temp)):
        if temp[i]:
            log[i]=''

    # return between _returnvalue_name_datarecord_f_b_returnvalue_ (new - remove if it does not work)
    #log=[re.sub('datarecord.*returnvalue', 'datarecord returnvalue', log[i]) for i in range(len(log))]

    
    # whitespace to underscore
    log=[re.sub(' ','_',log[i]) for i in range(len(log))]
    
    for i in range(len(log)):
        if log[i]=='':
            log[i]='empty_line'

   
    
    return log

###########################################################################################
############# PREPROCESSING ON A WORD LEVEL
###########################################################################################





###########################################################################################
############# CREATE DICTIONARY, INVERSE DICTIONARY AND COUNTER
###########################################################################################

def create_dictionary(preprocessed_log_lines_series):
    counter = collections.Counter()
    cleaned_tokens=list(itertools.chain(*preprocessed_log_lines_series))
    
    # one unknown token, this is needed because we may encounter an unknown token when trying a new file
    unknown_token = '<unk>'

    # word2index is the dictionary mapping a log key to an integer, each entry is of the format "log key: integer"
    word2index = {unknown_token: 0}

    # index2word is a list that returns a log key when an integer is entered (the "reverse" of word2index)
    index2word = [unknown_token]

    # counter counts all occurences, may be useful
    counter = Counter(cleaned_tokens)

    # below we iterate through the counter and add to index2word and word2index
    for word, count in counter.items():
        index2word.append(word)
        word2index[word] = len(word2index)

    num_classes = len(word2index)
    print('vocabulary size: ', num_classes)
    return word2index, index2word, counter


###########################################################################################
############# CREATE EVENTSEQUENCES
###########################################################################################

def text2sequence(cleaned_log, word2id):
    sequence=[word2id.get(cleaned_log[i],0) for i in range(len(cleaned_log))]
    return sequence


###########################################################################################
############# CREATE WINDOWS
###########################################################################################

def create_windows_targets(log_key_sequence, window_size, step):
    windows=[]
    targets=[]
    for i in range(0, len(log_key_sequence) - window_size, step):
        sentence = log_key_sequence[i:i + window_size]
        next_word = log_key_sequence[i + window_size]
        windows.append(sentence)
        targets.append(next_word)
    return (windows,targets)

###########################################################################################
############# CREATE INPUT TO MODELS: LISTS
###########################################################################################

def create_input(df, window_size, step):
    files=[]
    for i in range(len(df)):
        file=df.iloc[i].EventSequence
        files.append(file)

    windows=[]
    targets=[]
    for item in files:
        XX,yy=create_windows_targets(item, window_size, step)
        windows.extend(XX)
        targets.extend(yy)

    return windows, targets


###########################################################################################
############# CREATE INPUT TO MODELS: GENERATOR
###########################################################################################

from keras.utils import to_categorical

def batch_generator_LM(windows, targets, batch_size,word2id_size):
    len_data=len(windows)
    
    # to start over again each epoch
    while True:
        batch_start=0
        batch_end=batch_size
        
        # yield batches
        while batch_start+batch_size<len_data:
            limit=min(batch_end,len_data)
            
            # keras needs the input to be an array, and the target to be one-hot encoded ("to_categorical")
            batch_windows=np.array(windows[batch_start:limit])
            batch_targets=to_categorical(targets[batch_start:limit], word2id_size)
            
            yield (batch_windows, batch_targets)
            
            batch_start+=batch_size
            batch_end += batch_size