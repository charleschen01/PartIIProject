import torch

# for nn
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

from BaseModel import BaseModel

class RelationEmbeddingModel(BaseModel):

    def __init__(self, **kwds):

        # **kwds refers to the parameters in the base model
        super().__init__(**kwds)

        # we now initialise the embeddings for the first layer
        # need to convert one hot high dimensional vector to embeddings

        # for PyTorch's Embedding, input is a list of indices, output is corresponding word embeddings
        # Embedding is a look up table that stores embedding of a fixed size
        self.entityEmbedding = nn.Embedding(self.numEntity, self.dimension)
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

        # we initialise the embeddings and normalise them
        k_root = self.dimension**0.5
        self.entityEmbedding.weight.data = torch.FloatTensor(self.numEntity, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.entityEmbedding.weight.data = F.normalize(self.entityEmbedding.weight.data)
        self.relationEmbedding.weight.data = torch.FloatTensor(self.numRelation, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.relationEmbedding.weight.data = F.normalize(self.relationEmbedding.weight.data)

    @abstractmethod
    def tripletScore(self, leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):
        pass

    def forward(self, x):

        # TODO: another major bit to edit is to add a regularisation term for DistMult when we are caluclating the final costs

        # we pass in indices for relation, entities as well as negative entities
        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices = x

        # if possible, we leverage GPU to calculate scores
        if self.GPU:
            leftEnIndices = leftEnIndices.to(torch.device('cuda:0'))
            rightEnIndices = rightEnIndices.to(torch.device('cuda:0'))
            relIndices = relIndices.to(torch.device('cuda:0'))
            negLeftEnIndices = negLeftEnIndices.to(torch.device('cuda:0'))
            negRightEnIndices = negRightEnIndices.to(torch.device('cuda:0'))

        # pass the data through our first layer to get a list of entity embeddings
        leftEnEmbeddings = self.entityEmbedding(leftEnIndices)   # we get a list of embeddings
        rightEnEmbeddings = self.entityEmbedding(rightEnIndices)
        negLeftEnEmbeddings = self.entityEmbedding(negLeftEnIndices)
        negRightEnEmbeddings = self.entityEmbedding(negRightEnIndices)

        # second layer
        # we get relation embedding and calculate loss

        relEmbeddings = self.relationEmbedding(relIndices)

        # calculate score for the original triplet
        # validScores is a list of scores
        validScores = self.tripletScore(leftEnEmbeddings, relEmbeddings, rightEnEmbeddings)

        # scores for triplets where left is negative
        negLeftScores = self.tripletScore(negLeftEnEmbeddings, relEmbeddings, rightEnEmbeddings)

        # scores for triplets where right is negative
        negRightScores = self.tripletScore(leftEnEmbeddings, relEmbeddings, negRightEnEmbeddings)

        costl = self.margincost(validScores, negLeftScores)
        costr = self.margincost(validScores, negRightScores)
        cost = costl + costr

        # TODO: add regularisation here
        if self.whoami == "DistMult":
            regularisation = 0.0001 * torch.norm(relEmbeddings, dim=1)
            cost += regularisation

        # size of cost is the number of triplets passed in
        # take average so we return a scalar value
        mean_cost = torch.mean(cost)

        # move the result back to CPU
        if self.GPU:
            mean_cost = mean_cost.to(torch.device('cpu'))

        return mean_cost


    # the two functions below are for evaluation
    def evaluateLeftScores(self, rel, right):

        # we need to return a list of scores over all the corrupted left entities

        leftIndices = torch.LongTensor([*range(self.numEntity)])
        relationIndices = torch.LongTensor([rel])
        rightIndices = torch.LongTensor([right])

        if self.GPU:
            leftIndices = leftIndices.to(torch.device('cuda:0'))
            relationIndices = relationIndices.to(torch.device('cuda:0'))
            rightIndices = rightIndices.to(torch.device('cuda:0'))

        leftEnEmbedding = self.entityEmbedding(leftIndices)
        relationEmbedding = self.relationEmbedding(relationIndices)
        rightEnEmbedding = self.entityEmbedding(rightIndices)

        return self.scoreFunc(leftEnEmbedding, relationEmbedding, rightEnEmbedding).detach()

    def evaluateRightScores(self, left, rel):

        # return a list of similarity scores over all possible right entities

        leftIndices = torch.LongTensor([left])
        relationIndices = torch.LongTensor([rel])
        rightIndices = torch.LongTensor([*range(self.numEntity)])

        if self.GPU:
            leftIndices = leftIndices.to(torch.device('cuda:0'))
            relationIndices = relationIndices.to(torch.device('cuda:0'))
            rightIndices = rightIndices.to(torch.device('cuda:0'))

        leftEnEmbedding = self.entityEmbedding(leftIndices)
        relationEmbedding = self.relationEmbedding(relationIndices)
        rightEnEmbedding = self.entityEmbedding(rightIndices)

        return self.scoreFunc(leftEnEmbedding, relationEmbedding, rightEnEmbedding).detach()