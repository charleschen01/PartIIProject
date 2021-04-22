import torch

# for nn
import torch.nn as nn
import torch.nn.functional as F

from BaseModel import BaseModel

class NewModel(BaseModel):

    # input a list of embeddings in the form of tensor, output a list of values in tensor as well
    def L2Norm(self, embedding):
        return torch.sqrt(torch.sum(torch.square(embedding), dim=1))

    def L1Norm(self, embedding):
        return torch.sum(torch.abs(embedding), dim=1)

    def __init__(self, simi, **kwargs):

        super().__init__(**kwargs)
        self.whoami = "NewModel"

        # TODO: check; we might want to use another simi design
        if simi == "L1":
            self.normFunc = self.L1Norm
        elif simi == "L2":
            self.normFunc = self.L2Norm
        else:
            # TODO: throw exception if we pass in an invalid simi function
            pass

        # initialise embeddings for the first layer

        # for Embedding, input is a list of indices, output is corresponding word embeddings
        # Embedding is a look up table that stores embedding of a fixed size

        self.predVec = nn.Embedding(self.numEntity, self.dimension)
        self.predBias = nn.Embedding(self.numEntity, 1)

        # we create embeddings for all relations, but will only be using the ones corresponding to translation
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

        # we initialise the vector, bias and relation embedding; only normalise vector embedding
        k_root = self.dimension**0.5
        self.predVec.weight.data = torch.FloatTensor(self.numEntity, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.predVec.weight.data = F.normalize(self.predVec.weight.data)  # default dimension to normalise is 1

        # we try initialising all bias to 0 instead of random values
        # self.predBias.weight.data = torch.FloatTensor(self.numEntity, 1).uniform_(-6.0/k_root, 6.0/k_root)
        self.predBias.weight.data = torch.zeros(self.numEntity, 1)

        self.relationEmbedding.weight.data = torch.FloatTensor(self.numRelation, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)

    def forward(self, x):

        # we pass in indices for relation, entities as well as negative entities
        # these indices are in the form of long tensor
        # group is an integer, denoting the relation group this batch belongs

        # TODO: we might want to change the name because we are using "synset" instead of "entity"

        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices, group = x

        if self.GPU:
            # use GPU
            leftEnIndices = leftEnIndices.to(torch.device('cuda:0'))
            rightEnIndices = rightEnIndices.to(torch.device('cuda:0'))
            relIndices = relIndices.to(torch.device('cuda:0'))
            negLeftEnIndices = negLeftEnIndices.to(torch.device('cuda:0'))
            negRightEnIndices = negRightEnIndices.to(torch.device('cuda:0'))

        # here we guarantee that the batch of extended triplets all belong to the same relation

        # pass the data through our first layer to get a list of embeddings
        # the list of embeddings come in the form of tensor as well
        leftEnVec = self.predVec(leftEnIndices)
        leftEnBias = self.predBias(leftEnIndices)
        rightEnVec = self.predVec(rightEnIndices)
        rightEnBias = self.predBias(rightEnIndices)

        negLeftEnVec = self.predVec(negLeftEnIndices)
        negLeftEnBias = self.predBias(negLeftEnIndices)
        negRightEnVec = self.predVec(negRightEnIndices)
        negRightEnBias = self.predBias(negRightEnIndices)

        # second layer
        # we pass in the embedding list tensor through our second layer to calculate scores

        # calculate score for the original triplet
        # validScores is now a list of numbers
        validScores = self.tripletScore(leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias, group)

        # score for the triplets where left is negative
        negLeftScores = self.tripletScore(negLeftEnVec, negLeftEnBias, relIndices, rightEnVec, rightEnBias, group)

        # score for the triplets where right is negative
        negRightScores = self.tripletScore(leftEnVec, leftEnBias, relIndices, negRightEnVec, negRightEnBias, group)

        costl = self.margincost(validScores, negLeftScores)
        costr = self.margincost(validScores, negRightScores)
        cost = costl + costr

        # DEBUG: to find nan value in predicate embeddings
        # for i, x in enumerate(cost):
        #     if math.isnan(x):
        #         print("Found nan in %s" % i)
        #         # figure out what happens at i
        #         print("triplet crt score: %s" % crt[i])
        #         print("left neg crt score: %s" % crtln[i])
        #         print("right neg crt score: %s" % crtrn[i])
        #         print("left entity index: %s" % leftEnIndices[i])
        #         print("left entity vec: %s" % leftEnVec[i])
        #         print("left entity bias: %s" % leftEnBias[i])
        #         print("relation index: %s" % relIndices[i])
        #         print("right entity index: %s" % rightEnIndices[i])
        #         print("right entity vec: %s" % rightEnVec[i])
        #         print("right entity bias: %s" % rightEnBias[i])
        #         print("neg left entity vec: %s" % negLeftEnVec[i])
        #         print("neg left entity bias: %s" % negLeftEnBias[i])
        #         print("neg right entity vec: %s" % negRightEnVec[i])
        #         print("neg right entity bias: %s" % negRightEnBias[i])
        #
        #         # check if there's any problem with bias
        #         all_indices = torch.LongTensor([i for i in range(self.numEntity)])
        #         print(torch.sum(self.predBias(all_indices)))
        #         exit()

        # size of cost is the number of triplets passed in
        # we take average so the final output is only one value
        mean_cost = torch.mean(cost)

        if self.GPU:
            mean_cost = mean_cost.to(torch.device('cpu'))

        return mean_cost

    # this is the key part of our model
    def tripletScore(self, leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias, group):
        # the first five inputs are list of embeddings in the form of Tensor
        # output: a list of correctness scores, in the form of Tensor
        # we are only using relation embeddings when we are in the context shift case

        if group == 0:
            # hyponym
            # A <-> B; B is a hyponym of A, A is more general than B

            # if pixie is a normalised vector
            # (|v1-v2|L2 - (a1-a2))+
            vecDiff = self.normFunc(leftEnVec - rightEnVec)
            biasDiff = torch.reshape(leftEnBias - rightEnBias, (-1,))
            # -1 means the shape is inferred
            rawScore = vecDiff - biasDiff
            return - rawScore * (rawScore > 0)

            # TODO: implement the other case
            # trying a new method
            # s is sum of all (v1-v2) negative terms
            # ((a2-a1)-s)+
            # diff = leftEnVec - rightEnVec
            # negArr = torch.clamp(diff,max=0)
            # biasDiff = torch.reshape(rightEnBias - leftEnBias, (-1,))
            # rawScore = biasDiff - torch.sum(negArr,dim=1)
            # return rawScore * (rawScore > 0)

        elif group == 1:
            # hypernym
            # A <-> B; B is a hypernym of A, B is more general than A
            # (|v1-v2|L2 - (a2-a1))+
            vecDiff = self.normFunc(leftEnVec - rightEnVec)
            biasDiff = torch.reshape(rightEnBias - leftEnBias, (-1,))
            rawScore = vecDiff - biasDiff
            return - rawScore * (rawScore > 0)

            # TODO: implement the other case
            # trying a new method
            # calculate sum of all (v1-v2) positive terms
            # ((diff-(a2-a1))+
            # diff = leftEnVec - rightEnVec
            # posArr = torch.clamp(diff,min=0)
            # biasDiff = torch.reshape(rightEnBias - leftEnBias, (-1,))
            # rawScore = torch.sum(posArr, dim=1) - biasDiff
            # return rawScore * (rawScore > 0)

        elif group == 2:
            # synonym
            # |v1-v2|L2 + |a1-a2|
            vecDiff = self.normFunc(leftEnVec - rightEnVec)
            biasDiff = torch.abs(torch.reshape(leftEnBias - rightEnBias, (-1,)))
            return -(vecDiff + biasDiff)

            # DistMult approach:
            # relEmbedding = self.relationEmbedding(relIndices)
            # matrixProduct = torch.sum(leftEnVec * relEmbedding * rightEnVec, dim=1)
            # regularisation = 0.0001 * torch.norm(relEmbedding, dim=1)
            # return (matrixProduct + regularisation) * 0.2

        else:
            # context shift
            # |v1+d-v2|L2
            relEmbedding = self.relationEmbedding(relIndices)
            return -self.normFunc(leftEnVec + relEmbedding - rightEnVec)

    # the two functions below are used for evaluation
    def evaluateLeftScores(self, rel, right):

        # we need to return a list of scores over all possible left entities, in the form of Tensor
        # this operation is fast, because rel is fixed, and we can do batch calculation

        leftIndices = torch.LongTensor([*range(self.numEntity)])
        relationIndex = torch.LongTensor([rel])
        rightIndex = torch.LongTensor([right])

        if self.GPU:
            leftIndices = leftIndices.to(torch.device('cuda:0'))
            relationIndex = relationIndex.to(torch.device('cuda:0'))
            rightIndex = rightIndex.to(torch.device('cuda:0'))

        # we use LongTensor to store the indices
        # we are still calculating scores for all entities
        # but for corrupted entities, we will assign their scores to be inf later
        leftEnVec = self.predVec(leftIndices)
        leftEnBias = self.predBias(leftIndices)
        relationIndex = relationIndex # we only pass in one relation index because relation is fixed
        rightEnVec = self.predVec(rightIndex) # similarly, one right synset index
        rightEnBias = self.predBias(rightIndex)

        group = self.relGroup(rel)

        # it's okay to pass in single relation index and single synset embedding
        # torch will do the broadcasting for us
        return self.tripletScore(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group).detach()

    def evaluateRightScores(self, left, rel):

        # return a list of correctness scores over all possible right entities, in the form of Tensor

        leftIndex = torch.LongTensor([left])
        relationIndex = torch.LongTensor([rel])
        rightIndices = torch.LongTensor([*range(self.numEntity)])

        if self.GPU:
            leftIndex = leftIndex.to(torch.device('cuda:0'))
            relationIndex = relationIndex.to(torch.device('cuda:0'))
            rightIndices = rightIndices.to(torch.device('cuda:0'))

        leftEnVec = self.predVec(leftIndex)
        leftEnBias = self.predBias(leftIndex)
        relationIndex = relationIndex
        rightEnVec = self.predVec(rightIndices)
        rightEnBias = self.predBias(rightIndices)

        group = self.relGroup(rel)

        return self.crtFunc(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group).detach()