import torch

# for nn
import torch.nn as nn
import torch.nn.functional as F


# input a list of embeddings in the form of tensor, output a list of values in tensor as well
def L2Norm(embedding):
    return torch.sqrt(torch.sum(torch.square(embedding),dim=1))

def L1Norm(embedding):
    return torch.sum(torch.abs(embedding),dim=1)


class NewModel(nn.Module):

    def __init__(self, numEntity, numRelation, margin, simi, dimension, dataset, GPU):
        super(NewModel, self).__init__()

        self.numEntity = numEntity
        self.numRelation = numRelation
        self.dimension = dimension
        self.margin = margin
        self.GPU = GPU

        if simi == "L1":
            self.simiFunc = L1Norm
        elif simi == "L2":
            self.simiFunc = L2Norm

        # first layer: embedding layer
        # need to convert one hot high dimensional vector to embeddings

        # for Embedding, input is a list of indices, output is corresponding word embeddings
        # Embedding is a look up table that stores embedding of a fixed size

        self.predVec = nn.Embedding(self.numEntity, self.dimension)
        self.predBias = nn.Embedding(self.numEntity, 1)
        # we create embeddings for all relations, but will only be using the ones corresponding to translation
        self.relationEmbedding = nn.Embedding(self.numRelation, self.dimension)

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
        self.translation_set = {2, 7, 8, 9, 10, 11, 12, 14, 15, 16} if dataset == 'wn18' else {2, 5, 6, 7, 9}

        # we initialise the vector, bias and relation embedding; normalise vector and relation embedding
        k_root = dimension**0.5
        self.predVec.weight.data = torch.FloatTensor(self.numEntity, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)

        # try not to normalise the weight vector
        # self.predVec.weight.data = F.normalize(self.predVec.weight.data)  # default dimension to normalise is 1

        # we try initialising all bias to 0 because that makes more sense
        # self.predBias.weight.data = torch.FloatTensor(self.numEntity, 1).uniform_(-6.0/k_root, 6.0/k_root)
        self.predBias.weight.data = torch.zeros(self.numEntity, 1)
        # for bias, we don't normalise it because it's just one value
        self.relationEmbedding.weight.data = torch.FloatTensor(self.numRelation, self.dimension).uniform_(-6.0/k_root, 6.0/k_root)

        # try not to normalise the relation vector
        # self.relationEmbedding.weight.data = F.normalize(self.relationEmbedding.weight.data)

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


        # size of cost is the size of the list of entities/relations
        mean_cost = torch.mean(cost)
        if self.GPU:
            mean_cost = mean_cost.to(torch.device('cpu'))
        return mean_cost
        # we take average so the final output is only one value

    # this is the key part of our model
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
            vecDiff = self.simiFunc(leftEnVec - rightEnVec)
            biasDiff = torch.reshape(rightEnBias - leftEnBias, (-1,))
            rawScore = vecDiff - biasDiff
            return rawScore * (rawScore > 0)

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

    # the two functions below are used for evaluation
    def getLeftCrtScores(self, rel, right):

        # we need to return a list of correctness scores over all possible left entities, in the form of Tensor
        # this operation is fast, because rel is fixed, and we can do batch calculation

        leftIndices = torch.LongTensor([*range(self.numEntity)])
        relationIndex = torch.LongTensor([rel])
        rightIndices = torch.LongTensor([right])

        if self.GPU:
            leftIndices = leftIndices.to(torch.device('cuda:0'))
            relationIndex = relationIndex.to(torch.device('cuda:0'))
            rightIndices = rightIndices.to(torch.device('cuda:0'))

        # we use LongTensor to store the indices
        # we are still calculating scores for all entities
        # but for corrupted entities, we will assign their scores to be inf later
        leftEnVec = self.predVec(leftIndices)
        leftEnBias = self.predBias(leftIndices)
        relationIndex = relationIndex
        rightEnVec = self.predVec(rightIndices)
        rightEnBias = self.predBias(rightIndices)

        group = self.relGroup(rel)

        return self.crtFunc(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group).detach()

    def getRightCrtScores(self, left, rel):

        # return a list of correctness scores over all possible right entities, in the form of Tensor

        leftIndices = torch.LongTensor([left])
        relationIndex = torch.LongTensor([rel])
        rightIndices = torch.LongTensor([*range(self.numEntity)])

        if self.GPU:
            leftIndices = leftIndices.to(torch.device('cuda:0'))
            relationIndex = relationIndex.to(torch.device('cuda:0'))
            rightIndices = rightIndices.to(torch.device('cuda:0'))

        leftEnVec = self.predVec(leftIndices)
        leftEnBias = self.predBias(leftIndices)
        relationIndex = relationIndex
        rightEnVec = self.predVec(rightIndices)
        rightEnBias = self.predBias(rightIndices)

        group = self.relGroup(rel)

        return self.crtFunc(leftEnVec, leftEnBias, relationIndex, rightEnVec, rightEnBias, group).detach()