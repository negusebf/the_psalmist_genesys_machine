# Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LambdaCallback, TensorBoard, EarlyStopping, Callback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, Embedding, Bidirectional, GRU, LSTM, Conv1D, MaxPooling1D
import numpy as np
import random
import sys
import io
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import regularizers

# Define your global variables here



def num_to_book(book_name):
    book = {}
    book['genesis'] = 1
    book['exodus'] = 2
    book['leviticus'] = 3
    book['numbers'] = 4
    book['deutronomy'] = 5
    book['joshua'] = 6
    book['judges'] = 7
    book['ruth'] = 8
    book['1samuel'] = 9
    book['2samuel'] = 10
    book['1kings'] = 11
    book['2kings'] = 12
    book['1chronicles'] = 13
    book['2chronocles'] = 14
    book['ezra'] = 15
    book['nehemiah'] = 16
    book['esther'] = 17
    book['job'] = 18
    book['psalms'] = 19
    book['proverbs'] = 20
    book['ecclesiastes'] = 21
    book['song of solomon'] = 22
    book['isaiah'] = 23
    book['jeremiah'] = 24
    book['lamentations'] = 25
    book['ezekiel'] = 26
    book['daniel'] = 27
    book['hosea'] = 28
    book['joel'] = 29
    book['amos'] = 30
    book['obadiah'] = 31
    book['jonah'] = 32
    book['micah'] = 33
    book['nahum'] = 34
    book['habakkuk'] = 35
    book['zephaniah'] = 36
    book['haggai'] = 37
    book['zechariah'] = 38
    book['malachi'] = 39
    book['mathew'] = 40
    book['mark'] = 41
    book['luke'] = 42
    book['john'] = 43
    book['acts'] = 44
    book['romans'] = 45
    book['1corinthians'] = 46
    book['2corinthians'] = 47
    book['galatians'] = 48
    book['ephesians'] = 49
    book['philippians'] = 50
    book['colossians'] = 51
    book['1thessalonians'] = 52
    book['2thessalonians'] = 53
    book['1timothy'] = 54
    book['2timothy'] = 55
    book['titus'] = 56
    book['philemon'] = 57
    book['hebrews'] = 58
    book['james'] = 59
    book['1peter'] = 60
    book['2peter'] = 61
    book['1john'] = 62
    book['2john'] = 63
    book['3john'] = 64
    book['jude'] = 65
    book['revelation'] = 66
    return book[book_name]


def start_stop_indexer(book_name, book):
    df = book.drop(['id','v'], axis=1)
    start_index = 0
    stop_index = 0
    for i in range(len(df)):
        if df['b'].values[i] == num_to_book(book_name):
            start_index = i
            print("start_index = "+str(start_index))
            break

    for j in range(len(df)):
        if df['b'].values[j] != num_to_book(book_name) and df['b'].values[j] > num_to_book(book_name):
            stop_index = j
            print("stop_index = "+str(stop_index))
            break
    
    return (start_index, stop_index)
    

    
def build_data(path_to_file, corpus_type, select_book, tokenizer_name, verbose):
    print("Loading data...")
    print("Preprocessing data...")
    print("This may take a while depending on your corpus size...\n")
    book = pd.read_csv(path_to_file)
    # text
    start_index, stop_index = start_stop_indexer(select_book, book)
    book = book[start_index:stop_index]
    book_sentences = []
    if corpus_type == "verse":
        book = book.drop(["id", "b", "c", "v"], axis=1)
        # book.iloc[2,:].values[0]
        for i in range(len(book)):
            book_sentences.append(book.iloc[i,:].values[0])
    elif corpus_type == "chapter":
        book = book.drop(["id", "b", "v"], axis=1)
        # book.iloc[2,:].values[0]
        num_chapters = max(book['c'])
        for chaps in range(1, num_chapters+1):
            for i in range(1, len(book)+1):
                if book.values[i-1][0] == chaps:
                    book_sentences.append(book.values[i-1][1])
            book_sentences.append("\n")
        all_the_book = ""
        for sent in book_sentences:
            all_the_book += " " + str(sent)
        book_sentences = all_the_book
        book_sentences = book_sentences.lower().split("\n")
       
    # book_sentences


    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(book_sentences)
    word_index = tokenizer.word_index
    # word_index
    book_sequences = tokenizer.texts_to_sequences(book_sentences)  # Unecessary! We did it in the next cell. Just for checking.
    input_sequences = []
    for line in book_sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    # input_sequences
    max_sequence_len = max([len(x) for x in input_sequences]) # Take half of the longest sentence which is 64/2 in book
    book_padded = pad_sequences(book_sequences, max_sequence_len) # Unecessary! We did it in the next cell. Just for checking.
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # input_sequences
    # book_sentences[0]
    # book_sequences[0]
    # book_padded[0]
    # word_index['beginning']
    X = input_sequences[:,:-1]
    labels = input_sequences[:,-1]
    total_words = len(word_index) + 1
    Y = to_categorical(labels, num_classes=total_words)
    # if verbose == 1:
    #    print("One example of input_sequence "+str(input_sequences[0]))
    #    print("its corresponding X value is: "+str(X[0]))
    #    print("its corresponding label is: "+str(labels[0]))
    #    print("its corresponding Y (one_hot) value is: \n"+str(Y))
    #save_dict = word_index
    #json.dump(save_dict, open("psalms_word_indexer.txt",'w'))
    tokenizer_json = tokenizer.to_json()
    if tokenizer_name:
        tokenizer_name = tokenizer_name
    else:
        tokenizer_name = "word_tokenizer"
    with io.open(str(tokenizer_name)+str(".json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    if verbose == 1:
        print("\nOutput Summary:")
        print("Total words found: "+str(total_words))
        print("Generated sentence/sequence list length: "+str(len(book_sentences)))
        print("Indexed word list length: "+str(len(word_index)))
        print("Padded sequences numpy array shape: "+str(input_sequences.shape)+"\n\n")
    return X, Y, total_words, max_sequence_len



def build_model(path_to_file,
                corpus_type,
                select_book,
                tokenizer_name,
                model_save, 
                use_prev_model, 
                save_as, 
                loaded_model, 
                verbose=0,
                epochs=1):
    X, Y, total_words, max_sequence_len = build_data(path_to_file, corpus_type, select_book, tokenizer_name, verbose)
    print("Data preprocessing successfully completed...\n")
    class myCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.999):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
    
    my_callbacks = myCallback()
    
    if use_prev_model == True:
        print("Loading previously trained weights...")
        model = load_model(loaded_model)
        print("Training model using pre-trained weights...")
    else:
        model = Sequential()
        model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
        model.add(Bidirectional(LSTM(150, return_sequences = True)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        print("Training model from scratch...\n")
        print("\n")
    model.fit(X, Y, epochs=epochs) # You could use callbacks
    prev_hist = pd.read_csv('hist_chap.csv') #hist.csv
    hist = pd.DataFrame(model.history.history)
    new_hist = prev_hist.append(hist, ignore_index=True)
    new_hist.to_csv('hist_chap.csv', index=False) #hist.csv
    model.save(save_as)
    


def generate_output(next_words, token_file, model_used):
    max_sequence_len = 32
    #generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    usr_input = input("Your input: ")
    print("\n")
    # zero pad the sentence to Tx characters.
    #sentence = str(usr_input)
    sys.stdout.write("Generating Psalmist word...\n")
    print("\n")
    #print("You: "+str(str(usr_input)))
    #print("Machine: "+str(sentence))
    with open(token_file) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([usr_input])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        #print(token_list)
        model = load_model(model_used)
        predicted = model.predict_classes(token_list, verbose=0)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        usr_input += " " + output_word
    sys.stdout.write("Genesys-Machine: " + '"' + usr_input + '"')
    
    

#     sys.stdout.write("\n\nHere is your predicted words: \n\n") 
#     sys.stdout.write(usr_input)
#     for i in range(30):

#         x_pred = np.zeros((1, max_sequence_len, len(chars)))

#         for t, char in enumerate(sentence):
#             if char != '0':
#                 x_pred[0, t, char_indices[char]] = 1.

#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, temperature = 1.0)
#         next_char = indices_char[next_index]

#         generated += next_char
#         sentence = sentence[1:] + next_char

#         sys.stdout.write(next_char)
#         sys.stdout.flush()

#         if next_char == '\n':
#             continue
    