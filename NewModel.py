import torch
import numpy as np
import math

# for nn
import torch.nn as nn
import torch.nn.functional as F


# input a list of embeddings in the form of tensor, output a list of values in tensor as well
def L2Norm(embedding):
    return torch.sqrt(torch.sum(torch.square(embedding),dim=1))

def L1Norm(embedding):
    return torch.sum(torch.abs(embedding),dim=1)


class NewModel(nn.Module):

    def __init__(self, numEntity, numRelation, margin, simi, dimension):
        super(NewModel, self).__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin

        if simi == "L1":
            self.simiFunc = L1Norm
        elif simi == "L2":
            self.simiFunc = L2Norm

        # {'_also_see': 0, '_derivationally_related_form': 1, '_has_part': 2, '_hypernym': 3, '_hyponym': 4,
        #  '_instance_hypernym': 5, '_instance_hyponym': 6, '_member_holonym': 7, '_member_meronym': 8,
        #  '_member_of_domain_region': 9, '_member_of_domain_topic': 10, '_member_of_domain_usage': 11, '_part_of': 12,
        #  '_similar_to': 13, '_synset_domain_region_of': 14, '_synset_domain_topic_of': 15,
        #  '_synset_domain_usage_of': 16, '_verb_group': 17}
        self.hyponym_set = {4, 6}
        self.hypernym_set = {3, 5}
        self.synonym_set = {0, 1, 13, 17}
        self.translation_set = {2, 7, 8, 9, 10, 11, 12, 14, 15, 16}

        # first layer: embedding layer
        # need to convert one hot high dimensional vector to embeddings

        # for Embedding, input is a list of indices, output is corresponding word embeddings
        # Embedding is a look up table that stores embedding of a fixed size

        self.predVec = nn.Embedding(self.numEntity, self.dimension)
        self.predBias = nn.Embedding(self.numEntity, 1)
        # we create embeddings for all relations, but will only be using the ones corresponding to translation
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

        # we initialise the vector, bias and relation embedding; normalise vector and relation embedding
        k_root = dimension**0.5
        self.predVec.weight.data = torch.FloatTensor(self.numEntity, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.predVec.weight.data = F.normalize(self.predVec.weight.data)  # default dimension to normalise is 1
        self.predBias.weight.data = torch.FloatTensor(self.numEntity, 1).uniform_(-6.0/k_root, 6.0/k_root)
        # for bias, we don't normalise it because it's just one value
        self.relationEmbedding.weight.data = torch.FloatTensor(self.numRelation, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)
        self.relationEmbedding.weight.data = F.normalize(self.relationEmbedding.weight.data)

        # second layer: compute cost

        # input: list of embeddings for left entity, right entity, relation, negative left entity, negative right entity
        # output: average cost of the list

        # we define the second layer in the forward function because there's no available function to use

    # a function to decide which group a relation belongs
    def relGroup(self, i):
        if i in self.hyponym_set:
            return 0
        elif i in self.hypernym_set:
            return 1
        elif i in self.synonym_set:
            return 2
        else:
            # translation
            return 3

    def forward(self, x):

        # we pass in indices for relation, entities as well as negative entities
        # these indices are in the form of long tensor
        # group is an integer, denoting the relation group this batch belongs
        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices, group = x

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

        # now we pass the embedding list tensor through our second layer to calculate costs

        # correctness score for the original triplet
        # crt is now a list of numbers
        crt = self.crtFunc(leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias, group)

        # correctness score for the triplets where left is negative
        crtln = self.crtFunc(negLeftEnVec, negLeftEnBias, relIndices, rightEnVec, rightEnBias, group)

        # correctness score for the triplets where right is negative
        crtrn = self.crtFunc(leftEnVec, leftEnBias, relIndices, negRightEnVec, negRightEnBias, group)

        costl = self.margincost(crt, crtln, self.margin)
        costr = self.margincost(crt, crtrn, self.margin)
        cost = costl + costr

        # DEBUG: to find nan value in predicate embeddings
        for i, x in enumerate(cost):
            if math.isnan(x):
                print("Found nan in %s" % i)
                # figure out what happens at i
                print("triplet crt score: %s" % crt[i])
                print("left neg crt score: %s" % crtln[i])
                print("right neg crt score: %s" % crtrn[i])
                print("left entity index: %s" % leftEnIndices[i])
                print("left entity vec: %s" % leftEnVec[i])
                print("left entity bias: %s" % leftEnBias[i])
                print("relation index: %s" % relIndices[i])
                print("right entity index: %s" % rightEnIndices[i])
                print("right entity vec: %s" % rightEnVec[i])
                print("right entity bias: %s" % rightEnBias[i])
                print("neg left entity vec: %s" % negLeftEnVec[i])
                print("neg left entity bias: %s" % negLeftEnBias[i])
                print("neg right entity vec: %s" % negRightEnVec[i])
                print("neg right entity bias: %s" % negRightEnBias[i])

                # check if there's any problem with bias
                all_indices = torch.LongTensor([i for i in range(self.numEntity)])
                print(torch.sum(self.predBias(all_indices)))
                exit()


        # size of cost is the size of the list of entities/relations

        return torch.mean(cost)
        # we take average so the final output is only one value

    def crtFunc(self, leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias, group):
        # the first five inputs are list of embeddings in the form of Tensor
        # output: a list of correctness scores, in the form of Tensor
        # we are only using relation embeddings when we are in the translation case
        if group == 0:
            # hyponym
            # A <-> B; B is a hyponym of A, A is more general than B
            # (|v1-v2|L2 - (a1-a2))+
            vecDiff = self.simiFunc(leftEnVec - rightEnVec)
            biasDiff = torch.reshape(leftEnBias - rightEnBias, (-1,))
            # -1 means the shape is inferred
            rawScore = vecDiff - biasDiff
            return rawScore * (rawScore > 0)
        elif group == 1:
            # hypernym
            # A <-> B; B is a hypernym of A, B is more general than A
            # (|v1-v2|L2 - (a2-a1))+
            vecDiff = self.simiFunc(leftEnVec - rightEnVec)
            biasDiff = torch.reshape(rightEnBias - leftEnBias, (-1,))
            rawScore = vecDiff - biasDiff
            return rawScore * (rawScore > 0)
        elif group == 2:
            # synonym
            # |v1-v2|L1 + |a1-a2|
            vecDiff = self.simiFunc(leftEnVec - rightEnVec)
            biasDiff = torch.abs(torch.reshape(leftEnBias - rightEnBias, (-1,)))
            return vecDiff + biasDiff
        else:
            # translation
            # |v1+d-v2|L2
            relEmbedding = self.relationEmbedding(relIndices)
            return self.simiFunc(leftEnVec + relEmbedding - rightEnVec)

    def margincost(self, pos, neg, margin):
        out = pos - neg + margin
        # pos is supposed to be smaller, so we use pos - neg
        # print("pos scores are %s" % pos)
        # print("neg scores are %s" % neg)
        # print("costs are %s" % out)
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

            leftCrtScores = self.getLeftCrtScores(rel, right)

            # TODO: we probably want to check if there are many zeros here; bc zero is the perfect score
            # print("Across all left entities, number of 0 is %s" % (leftCrtScores == 0).sum())
            # this might indeed be a problem
            # after 10 epochs, out of 40943 entities, there are 1642 0s
            # after 20 epochs, out of 40943 entities, there are 39/25 0s
            # after 30 epochs, out of 40943 entities, there are 72/57 0s
            # after 50 epochs, out of 40943 entities, there are 45 0s

            # get ranking of idx from highest to lowest first (most correct to least correct)
            # for this list, items are idx and index is ranking
            # we then want to get the position of a particular index
            # use the where operation in numpy

            leftIdxRank = torch.argsort(leftCrtScores)
            # here we try to find out the rank of our left predicate
            leftRank = np.where(leftIdxRank == left)[0]
            leftRankList.extend(leftRank+1)

            # similarly, calculate rank of right
            rightCrtScores = self.getRightCrtScores(left, rel)
            # print("Across all right entities, number of 0 is %s" % (rightCrtScores == 0).sum())
            rightIdxRank = torch.argsort(rightCrtScores)
            rightRank = np.where(rightIdxRank == right)[0]
            rightRankList.extend(rightRank+1)

        return [leftRankList, rightRankList]

    def getLeftCrtScores(self, rel, right):

        # we need to return a list of correctness scores over all possible left entities, in the form of Tensor
        # this operation is fast, because rel is fixed, and we can do batch calculation

        # we use LongTensor to store the indices
        # leftEnVec and leftEnBias include all possible cases
        leftEnVec = self.predVec(torch.LongTensor([*range(self.numEntity)]))
        leftEnBias = self.predBias(torch.LongTensor([*range(self.numEntity)]))
        relationIndex = torch.LongTensor([rel])
        rightEnVec = self.predVec(torch.LongTensor([right]))
        rightEnBias = self.predBias(torch.LongTensor([right]))
        group = self.relGroup(rel)

        return self.crtFunc(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group)

    def getRightCrtScores(self, left, rel):

        # return a list of correctness scores over all possible right entities, in the form of Tensor

        rightEnVec = self.predVec(torch.LongTensor([*range(self.numEntity)]))
        rightEnBias = self.predBias(torch.LongTensor([*range(self.numEntity)]))
        relationIndex = torch.LongTensor([rel])
        leftEnVec = self.predVec(torch.LongTensor([left]))
        leftEnBias = self.predBias(torch.LongTensor([left]))
        group = self.relGroup(rel)

        return self.crtFunc(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group)