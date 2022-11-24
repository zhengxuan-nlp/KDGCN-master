# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from pytorch_pretrained_bert import BertTokenizer
from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

#bert
class Tokenizer_Bert:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        a = 0
        # text = text.lower()
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return sequence



def dependency_adj_matrix(text,start,end):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    #Check whether there is an aspect in the n hop
    for token in tokens:
        has_aspect = False
        matrix[token.i][token.i] = 1
        for child1 in token.children:
            #In the case of aspect, the syntax tree is preserved
            if token.i > start and token.i <= end:
                has_aspect = True
                break
            # non-aspect
            pos = child1.i
            if pos>start and pos<=end:
                has_aspect = True
                break
            for child2 in child1.children:
                pos = child2.i
                if pos > start and pos <= end:
                    has_aspect = True
                    break
                for child3 in child2.children:
                    pos = child3.i
                    if pos > start and pos <= end:
                        has_aspect = True
                        break
                    for child4 in child3.children:
                        pos = child4.i
                        if pos > start and pos <= end:
                            has_aspect = True
                            break

        if has_aspect:
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.concat_graph2', 'wb')
    tokenizer = Tokenizer_Bert()
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
        bert_text = tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.text_to_sequence(text))
        text_left = tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.text_to_sequence(text_left))
        aspect = tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.text_to_sequence(aspect))
        text = " ".join(bert_text)
        start = len(text_left)
        end = len(text_left)+len(aspect)
        adj_matrix = dependency_adj_matrix(text,start,end)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

if __name__ == '__main__':

    process('datasets/semeval14/restaurant_train.raw')
    process('datasets/semeval14/restaurant_test.raw')
    process('datasets/semeval14/laptop_train.raw')
    process('datasets/semeval14/laptop_test.raw')
    process('datasets/semeval15/restaurant_train.raw')
    process('datasets/semeval15/restaurant_test.raw')
    process('datasets/semeval16/restaurant_train.raw')
    process('datasets/semeval16/restaurant_test.raw')