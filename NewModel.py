import torch
import numpy as np
import math

# for nn
import torch.nn as nn
import torch.nn.functional as F


# input a 1D tensor, output a 1D tensor
def L2Norm(tensor_1d):
    return torch.sqrt(torch.sum(torch.square(tensor_1d)))


class NewModel(nn.Module):

    def __init__(self, numEntity, numRelation, margin, dimension=20):
        super(NewModel, self).__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin

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

    def forward(self, x):

        # we pass in indices for relation, entities as well as negative entities
        # these indices are in the form of long tensor
        leftEnIndices, rightEnIndices, relIndices, negLeftEnIndices, negRightEnIndices = x

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
        crt = self.crtFunc(leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias)

        # print("Triplet crt score: %s" % crt)

        # correctness score for the triplet where left is negative
        crtln = self.crtFunc(negLeftEnVec, negLeftEnBias, relIndices, rightEnVec, rightEnBias)

        # print("Left neg crt score: %s" % crtln)

        # correctness score for the triplet where right is negative
        crtrn = self.crtFunc(leftEnVec, leftEnBias, relIndices, negRightEnVec, negRightEnBias)

        # print("Right neg crt score: %s" % crtrn)

        costl = self.margincost(crt, crtln, self.margin)
        costr = self.margincost(crt, crtrn, self.margin)
        cost = costl + costr

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

    # we return a tensor of correctness scores here
    def crtFunc(self, leftEnVec, leftEnBias, relIndices, rightEnVec, rightEnBias):
        crtScores = torch.empty(len(relIndices))
        for i in range(len(relIndices)):
            crtScores[i] = self.crtScore(leftEnVec[i], leftEnBias[i], relIndices[i], rightEnVec[i], rightEnBias[i])
        return crtScores

    # calculate correct score for one triplet, based on which relation we are dealing with
    # return a single positive score tensor
    # relIndex is a long tensor, others are embeddings
    def crtScore(self, leftVec, leftBias, relIndex, rightVec, rightBias):
        # hyponym
        # TODO: okay, we cannot really do this on tensor
        if relIndex.item() in self.hyponym_set:
            # A <-> B; B is a hyponym of A, A is more general than B
            # (|v1-v2|L2 - (a1-a2))+
            vecDiff = L2Norm(leftVec - rightVec)
            biasDiff = leftBias-rightBias
            rawScore = vecDiff - biasDiff
            return rawScore * (rawScore > 0)
        # hypernym
        elif relIndex.item() in self.hypernym_set:
            # A <-> B; B is a hypernym of A, B is more general than A
            # (|v1-v2|L2 - (a2-a1))+
            vecDiff = L2Norm(leftVec - rightVec)
            biasDiff = rightBias - leftBias
            rawScore = vecDiff - biasDiff
            return rawScore * (rawScore > 0)
        # synonym
        elif relIndex.item() in self.synonym_set:
            # |v1-v2|L1 + |a1-a2|
            vecDiff = L2Norm(leftVec - rightVec)
            biasDiff = torch.abs(leftBias-rightBias)
            return vecDiff + biasDiff
        # translation
        else:
            # |v1+d-v2|L2
            relEmbedding = self.relationEmbedding(relIndex)
            return L2Norm(leftVec+relEmbedding-rightVec)

    def margincost(self, pos, neg, margin):
        out = pos - neg + margin
        # pos is supposed to be smaller, so we use pos - neg
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

            # TODO: this bit might be time consuming, we might want to reimplement crt score function so it can do the
            # TODO: calculation altogether instead of looping through every single value

            leftCrtScores = self.getLeftCrtScores(rel, right)

            # TODO: we probably want to check if there are many zeros here; bc zero is the perfect score

            # get ranking of idx from highest to lowest first (most correct to least correct)
            # for this list, items are idx and index is ranking
            # we then want to get the position of a particular index
            # can achieve this by argsort the list again to obtain a list of rankings, sorted in ascending index
            # however, we are using a simpler and faster for loop here

            leftIdxRank = np.argsort(leftCrtScores)
            leftRank = 0
            for num, idx in enumerate(leftIdxRank):
                if idx == left:
                    leftRank = num
                    break
            leftRankList.append(leftRank+1)

            # calculate rank of right
            rightCrtScores = self.getRightCrtScores(left, rel)
            rightIdxRank = np.argsort(rightCrtScores)
            rightRank = 0
            for num, idx in enumerate(rightIdxRank):
                if idx == right:
                    rightRank = num
                    break
            rightRankList.append(rightRank+1)

        return [leftRankList, rightRankList]

    def getLeftCrtScores(self, rel, right):

        # we need to return a numpy array of correctness scores over all possible left entities
        # numpy array because we want to use np.sort() later to obtain ranking

        # we use LongTensor to store the indices
        # need to use
        leftEnVec = self.predVec(torch.LongTensor([*range(self.numEntity)]))
        leftEnBias = self.predBias(torch.LongTensor([*range(self.numEntity)]))
        relationIndex = torch.LongTensor([rel])
        rightEnVec = self.predVec(torch.LongTensor([right]))
        rightEnBias = self.predBias(torch.LongTensor([right]))
        return np.asarray([self.crtScore(leftEnVec[i], leftEnBias[i], relationIndex, rightEnVec, rightEnBias)
                           for i in range(self.numEntity)])

    def getRightCrtScores(self, left, rel):

        # return a numpy array of correctness scores over all possible right entities

        rightEnVec = self.predVec(torch.LongTensor([*range(self.numEntity)]))
        rightEnBias = self.predBias(torch.LongTensor([*range(self.numEntity)]))
        relationIndex = torch.LongTensor([rel])
        leftEnVec = self.predVec(torch.LongTensor([left]))
        leftEnBias = self.predBias(torch.LongTensor([left]))

        return np.asarray([self.crtScore(leftEnVec, leftEnBias, relationIndex, rightEnVec[i], rightEnBias[i])
                           for i in range(self.numEntity)])
