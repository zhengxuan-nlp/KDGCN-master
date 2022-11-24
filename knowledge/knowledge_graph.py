import sys
import numpy as np
from senticnet5 import senticnet
from utils import normalize
import pickle



def onehot(x, primary_mood):
    one_hot = np.zeros(len(primary_mood), dtype='float32')
    one_hot[primary_mood[x]] = 1
    return one_hot


def node2vec():
    filename = "./affectivespace.csv"
    fr = open(filename)
    word_vec = {}
    for line in fr.readlines():
        tokens = line.rstrip().split(",")
        word_vec[tokens[0]] = np.asarray(tokens[-100:], dtype='float32')
    return word_vec


class Graph():
    def __init__(self):
        self.graph = senticnet
        self.primary_mood_to_id = self.extract_primary_mood()
        self.secondary_mood_to_id = self.extract_secondary_mood()
        self.mood_to_id = self.extract_mood()
        self.polarity_label_to_id = self.extract_polarity_label()
        self.word2vec = node2vec()
        self.words = {}

    def extract_primary_mood(self):
        primary_mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][4]
            if tmp not in primary_mood:
                primary_mood[tmp] = num
                num += 1
        return primary_mood

    def extract_secondary_mood(self):
        secondary_mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][5]
            if tmp not in secondary_mood:
                secondary_mood[tmp] = num
                num += 1
        return secondary_mood

    def extract_mood(self):
        mood = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][4]
            if tmp not in mood:
                mood[tmp] = num
                num += 1
            tmp = self.graph[word][5]
            if tmp not in mood:
                mood[tmp] = num
                num += 1
        return mood

    def extract_polarity_label(self):
        polarity_label = {}
        num = 0
        for word in self.graph.keys():
            tmp = self.graph[word][6]
            if tmp not in polarity_label:
                polarity_label[tmp] = num
                num += 1
        return polarity_label

    def get_vec(self, word):
        if word not in self.word2vec:
            return np.zeros(100, dtype='float32')
        else:
            return self.word2vec[word]

    def infoextract(self, word):
        vec = self.get_vec(word)
        if word not in self.graph:
            return np.zeros(5 + len(self.mood_to_id) * 2 + len(self.polarity_label_to_id), dtype='float32'), vec, []
        data = self.graph[word]
        ans = []
        pleasantness_value = float(data[0])
        ans.append(pleasantness_value)          #pleasantness值
        attention_value = float(data[1])
        ans.append(attention_value)             #attention值
        sensitivity_value = float(data[2])
        ans.append(sensitivity_value)           #sensitivity_value
        aptitude_value = float(data[3])
        ans.append(aptitude_value)              #aptitude_value
        primary_mood = data[4]
        primary_mood_onehot = onehot(primary_mood, self.mood_to_id)   #primary_mood_onehot
        ans += list(primary_mood_onehot)
        secondary_mood = data[5]
        secondary_mood_onehot = onehot(secondary_mood, self.mood_to_id)#secondary_mood_onehot
        ans += list(secondary_mood_onehot)
        polarity_label = data[6]
        polarity_label_onehot = onehot(polarity_label, self.polarity_label_to_id)#polarity_label_onehot
        ans += list(polarity_label_onehot)
        polarity_value = float(data[7])
        ans.append(polarity_value)

        # ans += list(vec)
        ans = np.array(ans, dtype='float32')
        semantics = []
        semantics1 = data[8]
        semantics.append(semantics1)#related words 1
        semantics2 = data[9]
        semantics.append(semantics2)#related words 2
        semantics3 = data[10]
        semantics.append(semantics3)#related words 3
        semantics4 = data[11]
        semantics.append(semantics4)#related words 4
        semantics5 = data[12]
        semantics.append(semantics5)#related words 5

        return ans, vec, semantics


def get_kg_feature(text, max_sequence_len=100):
    graph = Graph()
    words = text.split()
    features = np.zeros((max_sequence_len, 123), dtype='float32')

    for i in range(len(words)):
        feature, vec, semantics = graph.infoextract(words[i])
        # print(feature.shape, vec.shape)
        features[i] = np.concatenate((feature, vec), axis=0)
    features = np.array(features, dtype='float32')
    return features



def build_graph(text,max_sequence_len=100, max_node_num=30):
    graph = Graph()
    words = text.split()

    features = np.zeros((max_sequence_len+ max_node_num, 23), dtype='float32')
    for i in range(len(words)):
        # print("word::::::", words[i])
        feature, vec, semantics = graph.infoextract(words[i])
        features[i] = feature
    features = np.array(features, dtype='float32')
    return features

def build_graph_file(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    all_data = {}
    fout = open(fname + '.kgf', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + " " + aspect + " " + text_right
        features = build_graph(text)
        all_data[i] = [features]
    pickle.dump(all_data, fout)
    fout.close()

def get_aspect_expand(aspect,max_node_num=30):
    graph = Graph()
    aspect_expand = list()
    aspects = aspect.split()
    nodes_id = {}
    num = 0
    features = np.zeros((max_node_num,123), dtype='float32')
    for i in range(len(aspects)):
        feature, vec, semantics = graph.infoextract(aspects[i])
        # print(semantics)
        for node in semantics:
            if node not in nodes_id:
                aspect_expand.append(node)
                nodes_id[node] = num
                feature_node, vec_node, semantics_node = graph.infoextract(node)
                features[nodes_id[node]] = np.concatenate((feature_node, vec_node))
                num += 1
            if len(nodes_id) >= max_node_num:
                break
        if len(nodes_id) >= max_node_num:
            break

    return features, aspect_expand


def build_aspect_file(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    all_data = {}
    fout = open(fname + '.aspect', 'wb')
    for i in range(0, len(lines), 3):
        aspect = lines[i + 1].lower().strip()
        features, aspect_expand = get_aspect_expand(aspect)
        all_data[i] = [features,aspect_expand]
    pickle.dump(all_data, fout)
    fout.close()


def build_graph_feature_file(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    all_data = {}
    fout = open(fname + '.kgf', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + " " + aspect + " " + text_right
        adj, features = build_graph(text_left, text, aspect)
        all_data[i] = [adj, features]
    pickle.dump(all_data, fout)
    fout.close()



if __name__ == '__main__':




    build_aspect_file('../datasets/semeval15/restaurant_train.raw')
    # build_aspect_file('../datasets/semeval15/restaurant_test.raw')
    # build_aspect_file('../datasets/semeval16/restaurant_train.raw')
    # build_aspect_file('../datasets/semeval16/restaurant_test.raw')


    # build_graph_file('../datasets/semeval14/laptop_train.raw')
    # build_graph_file('../datasets/semeval14/laptop_test.raw')





