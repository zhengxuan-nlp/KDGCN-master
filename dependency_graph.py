# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

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
    fout = open(filename+'.graph2', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        start = len(text_left.split(' '))-1
        end = len(text_left.split(' '))+len(aspect.split(' ')) - 1
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, start, end)
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