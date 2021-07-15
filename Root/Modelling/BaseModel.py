import torch.nn as nn

from abc import abstractmethod

class BaseModel(nn.Module):

    def __init__(self, numEntity, numRelation, dimension, margin, dataset, GPU):

        super().__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin
        self.dataset = dataset
        self.GPU = GPU

        # declare relation categories
        # this is to evaluate the performance for different relations respectively

        # WN18:
        # {'_also_see': 0,
        #  '_derivationally_related_form': 1,
        #  '_has_part': 2,
        #  '_hypernym': 3,
        #  '_hyponym': 4,
        #  '_instance_hypernym': 5,
        #  '_instance_hyponym': 6,
        #  '_member_holonym': 7,
        #  '_member_meronym': 8,
        #  '_member_of_domain_region': 9,
        #  '_member_of_domain_topic': 10,
        #  '_member_of_domain_usage': 11,
        #  '_part_of': 12,
        #  '_similar_to': 13,
        #  '_synset_domain_region_of': 14,
        #  '_synset_domain_topic_of': 15,
        #  '_synset_domain_usage_of': 16,
        #  '_verb_group': 17}

        # WN18RR:
        # {0: '_also_see',
        #  1: '_derivationally_related_form',
        #  2: '_has_part',
        #  3: '_hypernym',
        #  4: '_instance_hypernym',
        #  5: '_member_meronym',
        #  6: '_member_of_domain_region',
        #  7: '_member_of_domain_usage',
        #  8: '_similar_to',
        #  9: '_synset_domain_topic_of',
        #  10: '_verb_group'}

        self.hyponym_set = {4, 6} if dataset == 'wn18' else {}
        self.hypernym_set = {3, 5} if dataset == 'wn18' else {3, 4}
        self.synonym_set = {0, 1, 13, 17} if dataset == 'wn18' else {0, 1, 8, 10}
        self.context_shift_set = {2, 7, 8, 9, 10, 11, 12, 14, 15, 16} if dataset == 'wn18' else {2, 5, 6, 7, 9}

        # we don't initialise embeddings in the base class yet

    def relGroup(self, i):
        if i in self.hyponym_set:
            return 0
        elif i in self.hypernym_set:
            return 1
        elif i in self.synonym_set:
            return 2
        elif i in self.context_shift_set:
            return 3
        else:
            raise ValueError("Invalid relation index!")

    def margincost(self, pos, neg):

        # we want pos to have a higher score than neg
        # so essentially we want to minimise (neg-pos)
        out = neg - pos + self.margin

        return out * (out > 0)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def tripletScore(self):
        pass

    # functions for evaluation
    # for these two functions, one's defined in relation embedding model, the other defined in new model
    @abstractmethod
    def evaluateSubjectScores(self, rel, right):
        pass

    @abstractmethod
    def evaluateObjectScores(self, left, rel):
        pass