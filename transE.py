import torch

# for nn
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):

    def __init__(self, numEntity, numRelation, dimension=50, margin=1.0):
        super(TransE, self).__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin

        # first layer: embedding layer
        # need to convert one hot high dimensional vector to embeddings

        # for Embedding, input is a list of indices, output is corresponding word embeddings
        self.entityEmbedding = nn.Embedding(self.numEntity, self.dimension)
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

        # we initialise the embeddings
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

        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices = x

        # pass the data through our first layer to get a list of embeddings
        leftEnEmbeddings = self.entityEmbedding(leftEnIndices)   # we get a list of embeddings
        rightEnEmbeddings = self.entityEmbedding(rightEnIndices)
        relEmbeddings = self.relationEmbedding(relIndices)
        negLeftEnEmbeddings = self.entityEmbedding(leftEnIndices)
        negRightEnEmbeddings = self.entityEmbedding(rightEnIndices)

        # normalise embeddings
        # essentially we normalise embeddings at the start of each new round
        leftEnEmbeddings = F.normalize(leftEnEmbeddings)
        rightEnEmbeddings = F.normalize(rightEnEmbeddings)
        relEmbeddings = F.normalize(relEmbeddings)
        negLeftEnEmbeddings = F.normalize(negLeftEnEmbeddings)
        negRightEnEmbeddings = F.normalize(negRightEnEmbeddings)

        # now we pass the embeddings through our second layer to calculate costs
        simi = torch.sum((leftEnEmbeddings + relEmbeddings)*rightEnEmbeddings, dim=1)
        # sum over all columns for each row
        similn = torch.sum((negLeftEnEmbeddings + relEmbeddings)*rightEnEmbeddings, dim=1)
        simirn = torch.sum((leftEnEmbeddings + relEmbeddings)*negRightEnEmbeddings, dim=1)
        costl = self.margincost(simi, similn, self.margin)
        costr = self.margincost(simi, simirn, self.margin)
        cost = costl + costr
        return torch.mean(cost)
        # size of output layer is the size of the list of entities/relations

    def margincost(self, pos, neg, margin):
        out = neg - pos + margin
        return out * (out > 0)

