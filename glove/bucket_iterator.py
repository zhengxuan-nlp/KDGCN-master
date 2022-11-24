# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

# glove
class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_dependency_tree = []
        batch_kg_feature = []
        batch_aspect_feature = []
        batch_aspect_expand = []

        batch_position = []
        # batch_conj = []
        max_len1 = max([len(t[self.sort_key]) for t in batch_data])
        max_len2 = max([len(t['position_tag']) for t in batch_data])
        max_len3 = max([len(t['aspect_expand']) for t in batch_data])
        max_len = max(max_len3,max(max_len1,max_len2))

        # max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, \
            dependency_tree, position_tag, kg_feature,aspect_feature,aspect_expand= \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['dependency_tree'], item['position_tag'],  item['kg_feature'],\
                item['aspect_feature'],item['aspect_expand']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            position_padding = [0] * (max_len - len(position_tag))
            aspect_expand_padding = [0] * (max_len - len(aspect_expand))


            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_position.append(position_tag + position_padding)
            batch_aspect_expand.append(aspect_expand + aspect_expand_padding)
            # batch_conj.append(conj)

            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_dependency_tree.append(numpy.pad(dependency_tree, \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))

            # batch_kg_feature.append(numpy.pad(kg_feature, \
            #     ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))

            batch_kg_feature.append(kg_feature)
            batch_aspect_feature.append(aspect_feature)

        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'dependency_tree': torch.tensor(batch_dependency_tree), \
                'position_tag': torch.tensor(batch_position),\
                'kg_feature':torch.tensor(batch_kg_feature), \
                'aspect_expand': torch.tensor(batch_aspect_expand),\
                'aspect_feature': torch.tensor(batch_aspect_feature)
            # 'conj':torch.tensor(batch_conj)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

