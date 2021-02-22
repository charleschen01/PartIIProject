import torch
import numpy as np

# for nn
import torch.nn as nn
import torch.nn.functional as F

# we define different similarity scores here


def L1Similarity(leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):

    return torch.sum(torch.abs(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1)


def L2Similarity(leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):

    return torch.sqrt(torch.sum(torch.square(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1))


def DotSimilarity(leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):

    # if we use dot similarity, the larger the score, the better the match

    # inputs: lists of embeddings for left entity, relation and right entity; in the form of tensor
    # output: list of similarity scores; in the form of tensor

    # for dot product, the larger the value, the more similar it is
    # we therefore need to take the negative so that smaller value corresponds to high similarity
    return -torch.sum((leftEnEmbeddings + relEmbeddings)*rightEnEmbeddings, dim=1)



class TransE(nn.Module):

    def __init__(self, numEntity, numRelation, simi, margin, dimension):
        super(TransE, self).__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin
        if simi == "L1":
            self.simifunc = L1Similarity
        elif simi == "L2":
            self.simifunc = L2Similarity
        else:
            self.simifunc = DotSimilarity


        # first layer: embedding layer
        # need to convert one hot high dimensional vector to embeddings

        # for Embedding, input is a list of indices, output is corresponding word embeddings
        # Embedding is a look up table that stores embedding of a fixed size
        self.entityEmbedding = nn.Embedding(self.numEntity, self.dimension)
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

        # we initialise the embeddings and normalise them
        k_root = dimension**0.5
        self.entityEmbedding.weight.data = torch.FloatTensor(self.numEntity, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.entityEmbedding.weight.data = F.normalize(self.entityEmbedding.weight.data)
        self.relationEmbedding.weight.data = torch.FloatTensor(self.numRelation, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.relationEmbedding.weight.data = F.normalize(self.relationEmbedding.weight.data)

        # second layer: compute cost

        # input: list of embeddings for left entity, right entity, relation, negative left entity, negative right entity
        # output: average cost of the list

        # we define the second layer in the forward function because there's no available function to use

    def forward(self, x):

        # we pass in indices for relation, entities as well as negative entities
        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices = x

        # pass the data through our first layer to get a list of embeddings
        leftEnEmbeddings = self.entityEmbedding(leftEnIndices)   # we get a list of embeddings
        rightEnEmbeddings = self.entityEmbedding(rightEnIndices)
        relEmbeddings = self.relationEmbedding(relIndices)
        negLeftEnEmbeddings = self.entityEmbedding(negLeftEnIndices)
        negRightEnEmbeddings = self.entityEmbedding(negRightEnIndices)

        # now we pass the embeddings through our second layer to calculate costs

        # here we use the dot product method

        # similarity score for the original triplet
        # sum over all columns for each row
        # simi is now a list of numbers
        simi = self.simifunc(leftEnEmbeddings, relEmbeddings, rightEnEmbeddings)

        # similarity score for the triplet where left is negative
        similn = self.simifunc(negLeftEnEmbeddings, relEmbeddings, rightEnEmbeddings)

        # similarity score for the triplet where right is negative
        simirn = self.simifunc(leftEnEmbeddings, relEmbeddings, negRightEnEmbeddings)

        costl = self.margincost(simi, similn, self.margin)
        costr = self.margincost(simi, simirn, self.margin)
        cost = costl + costr
        # size of cost is the size of the list of entities/relations
        return torch.mean(cost)
        # we take average so the final output is only one value

    def margincost(self, pos, neg, margin):
        out = pos - neg + margin
        # pos is more similar and supposed to be smaller, so we use pos - neg
        return out * (out > 0)

    def rankScore(self, idxLeft, idxRel, idxRight):

        # calculate predicted ranks for a particular state of embeddings

        # inputs: idxLeft, idxRight and idxRel are all lists of indices
        # outputs: a list [llist,rlist]; llist and rlist contain ranks of left entity and right entity respectively

        leftRankList = []
        rightRankList = []

        for left, rel, right in zip(idxLeft, idxRel, idxRight):
            # calculate rank of left
            # we need to calculate the similarity score of all 'left' entities given rel and right

            # both leftSimiScores and rightSimiScores are numpy arrays

            leftSimiScores = self.getLeftSimiScores(rel, right)

            # get ranking of idx from highest to lowest first
            # for this list, items are idx and index is ranking
            # we then want to get the position of a particular index
            # can achieve this by argsort the list again to obtain a list of rankings, sorted in ascending index

            leftIdxRank = np.argsort(leftSimiScores)
            leftRank = 0
            for num, idx in enumerate(leftIdxRank):
                if idx == left:
                    leftRank = num
                    break
            leftRankList.append(leftRank+1)

            # calculate rank of right
            rightSimiScores = self.getRightSimiScores(left, rel)
            rightIdxRank = np.argsort(rightSimiScores)
            rightRank = 0
            for num, idx in enumerate(rightIdxRank):
                if idx == right:
                    rightRank = num
                    break
            rightRankList.append(rightRank+1)

        return [leftRankList, rightRankList]

    def getLeftSimiScores(self, rel, right):

        # we need to return a list of similarity scores over all possible left entities

        leftEnEmbeddings = self.entityEmbedding(torch.LongTensor([i for i in range(self.entityEmbedding.num_embeddings)]))
        relationEmbedding = self.relationEmbedding(torch.LongTensor([rel]))
        rightEnEmbedding = self.entityEmbedding(torch.LongTensor([right]))
        return self.simifunc(leftEnEmbeddings, relationEmbedding, rightEnEmbedding).detach().numpy()

    def getRightSimiScores(self, left, rel):

        # return a list of similarity scores over all possible right entities

        rightEnEmbeddings = self.entityEmbedding(torch.LongTensor([i for i in range(self.entityEmbedding.num_embeddings)]))
        leftEnEmbedding = self.entityEmbedding(torch.LongTensor([left]))
        relationEmbedding = self.relationEmbedding(torch.LongTensor([rel]))
        return self.simifunc(leftEnEmbedding, relationEmbedding, rightEnEmbeddings).detach().numpy()



