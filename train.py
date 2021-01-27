# similar to WN_parse.py
# convert triplets to sparse matrices which can serve as inputs to the neural network
import scipy.sparse
import pickle
import numpy as np
from transE import TransE
from NewModel import NewModel
import random

# optimiser
import torch
import torch.optim as optim

import torch.nn.functional as F

import time
import math



def load_file(path):
    return scipy.sparse.csr_matrix(pickle.load(open(path, 'rb')), dtype='float32')


def convert2idx(spmat):

    # nonzero() returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]
    # return the indices of the non-zero elements, arranged in the order of triplets
    # a smart way to compress the sparse matrix
    # makes sure different datasets will maintain the same order
    # eventually we will be able to get [idx for entity in triplet 1, idx for entity in triplet 2, ...]


class train:

    def __init__(self):

        # load the 4 dictionaries for entities and relations

        # subset
        # entity2id = pickle.load(open("./data/wn18_subset_entity2id", 'rb'))
        # id2entity = pickle.load(open("./data/wn18_subset_id2entity", 'rb'))
        # id2relation = pickle.load(open("./data/wn18_subset_id2relation", 'rb'))
        # relation2id = pickle.load(open("./data/wn18_subset_relation2id", 'rb'))

        # full dataset
        entity2id = pickle.load(open("./data/wn18_entity2id", 'rb'))
        id2entity = pickle.load(open("./data/wn18_id2entity", 'rb'))
        id2relation = pickle.load(open("./data/wn18_id2relation", 'rb'))
        relation2id = pickle.load(open("./data/wn18_relation2id", 'rb'))

        self.numEntity = len(entity2id)
        self.numRelation = len(relation2id)  # should be 18 here
        # the original figure is 1000, 50 is just for testing purpose
        # TODO: need to adjust it for the actual training
        self.total_epoch = 1
        # divide our triplets into 100 batches
        self.nbatches = 100

        # we need to obtain indices for left entity, right entity, relation, as well as negative left and right entities
        # which we can then feed into the network

        # the format of datasets below is sparse csr files of size (no. of entities/relations * no. of triplets)

        # use subset

        # # positives
        # trainLeft = load_file("./data/wn18_subset_train_lhs")
        # trainRight = load_file("./data/wn18_subset_train_rhs")
        # trainRel = load_file("./data/wn18_subset_train_rel")
        #
        # # validation set
        # validLeft = load_file("./data/wn18_subset_valid_lhs")
        # validRight = load_file("./data/wn18_subset_valid_rhs")
        # validRel = load_file("./data/wn18_subset_valid_rel")
        #
        # # test set
        # testLeft = load_file("./data/wn18_subset_test_lhs")
        # testRight = load_file("./data/wn18_subset_test_rhs")
        # testRel = load_file("./data/wn18_subset_test_rel")

        # use full dataset

        trainLeft = load_file("./data/wn18_train_lhs")
        trainRight = load_file("./data/wn18_train_rhs")
        trainRel = load_file("./data/wn18_train_rel")

        # validation set
        validLeft = load_file("./data/wn18_valid_lhs")
        validRight = load_file("./data/wn18_valid_rhs")
        validRel = load_file("./data/wn18_valid_rel")

        # test set
        testLeft = load_file("./data/wn18_test_lhs")
        testRight = load_file("./data/wn18_test_rhs")
        testRel = load_file("./data/wn18_test_rel")

        # index conversion
        # these are the indices that we can directly feed into our first layer
        self.trainLeftIndex = convert2idx(trainLeft)
        self.trainRightIndex = convert2idx(trainRight)
        self.trainRelIndex = convert2idx(trainRel)
        self.validLeftIndex = convert2idx(validLeft)
        self.validRightIndex = convert2idx(validRight)
        self.validRelIndex = convert2idx(validRel)
        self.testLeftIndex = convert2idx(testLeft)
        self.testRightIndex = convert2idx(testRight)
        self.testRelIndex = convert2idx(testRel)

    def train(self, lr, margin):

        # initialise the neural network
        # TransE
        # transE = TransE(self.numEntity, self.numRelation, simi, margin)
        # newModel
        newModel = NewModel(self.numEntity, self.numRelation, margin)

        # start training process
        inputsize = len(self.trainLeftIndex)
        print("Input size is %s (number of triplets)" % inputsize)
        batchsize = inputsize//self.nbatches

        optimizer = optim.Adam(newModel.parameters(), lr)

        print("Start training...")
        print("Parameters: Learning rate: {0}, Margin: {1}".format(lr, margin))

        finalEpochLoss = 0

        for epoch_count in range(self.total_epoch):

            print("Start epoch %s" % epoch_count)

            epochloss = 0

            # shuffling
            order = np.random.permutation(inputsize)
            trainLeftIndex = self.trainLeftIndex[order]
            trainRightIndex = self.trainRightIndex[order]
            trainRelIndex = self.trainRelIndex[order]

            # negatives
            # we choose a random index
            trainNegLeftIndex = np.random.choice(self.numEntity, inputsize, replace=True)
            trainNegRightIndex = np.random.choice(self.numEntity, inputsize, replace=True)

            for i in range(self.nbatches):

                # TODO: maybe convert this manual process into a smarter PyTorch one using dataloader?

                # TODO: what about the data left behind that doesn't fit into one batchsize?

                # we use LongTensor because our embeddings expect long tensor inputs
                batchLeftIndex = torch.LongTensor(trainLeftIndex[i*batchsize:(i+1)*batchsize])
                batchRightIndex = torch.LongTensor(trainRightIndex[i*batchsize:(i+1)*batchsize])
                batchRelIndex = torch.LongTensor(trainRelIndex[i*batchsize:(i+1)*batchsize])
                batchNegLeftIndex = torch.LongTensor(trainNegLeftIndex[i*batchsize:(i+1)*batchsize])
                batchNegRightIndex = torch.LongTensor(trainNegRightIndex[i*batchsize:(i+1)*batchsize])

                x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex)
                newModel.zero_grad()
                loss = newModel(x)  # dynamically creating the computation graph
                loss.backward()
                optimizer.step()

                print("Batch {0} loss is {1}".format(i+1, loss))

                epochloss += loss

                if i % 10 == 9:
                    print("Finished batch {0}/{1}".format(i+1, self.nbatches))

                # check predicates to see if there's any nan value
                all_indices = torch.LongTensor([i for i in range(newModel.numEntity)])
                for j, x in enumerate(newModel.predVec(all_indices)):
                    if math.isnan(x[0]):
                        print("predicate %s has become nan!" % j)
                        # okay, we've found a nan; now we want to see what operation makes the predicate become nan
                        badBatchIndex = 0
                        for batchIdx in range(100):
                            if batchLeftIndex[batchIdx] == j or batchRightIndex[batchIdx] == j or batchNegLeftIndex[batchIdx] == j or batchNegRightIndex[batchIdx] == j:
                                badBatchIndex = batchIdx
                                break
                        if badBatchIndex == 0:
                            print("nan value not found in this batch")
                        # we've find the badBatchIndex
                        # now let's find all the training data there
                        leftIndex = batchLeftIndex[badBatchIndex]
                        print("left entity index: %s" % leftIndex)
                        leftVec = newModel.predVec(leftIndex)
                        print("left entity vec: %s" % leftVec)
                        leftBias = newModel.predBias(leftIndex)
                        print("left entity bias: %s" % leftBias)
                        relationIndex = batchRelIndex[badBatchIndex]
                        print("relation index: %s" % relationIndex)
                        print("relation embedding: %s" % newModel.relationEmbedding(relationIndex))
                        rightIndex = batchRightIndex[badBatchIndex]
                        print("right entity index: %s" % rightIndex)
                        rightVec = newModel.predVec(rightIndex)
                        print("right entity vec: %s" % rightVec)
                        rightBias = newModel.predBias(rightIndex)
                        print("right entity bias: %s" % rightBias)
                        negLeftIndex = batchNegLeftIndex[badBatchIndex]
                        print("negative left entity index: %s" % negLeftIndex)
                        negLeftVec = newModel.predVec(negLeftIndex)
                        print("neg left entity vec: %s" % negLeftVec)
                        negLeftBias = newModel.predBias(negLeftIndex)
                        print("neg left entity bias: %s" % negLeftBias)
                        negRightIndex = batchNegRightIndex[badBatchIndex]
                        print("negative right entity index: %s" % negRightIndex)
                        negRightVec = newModel.predVec(negRightIndex)
                        print("neg right entity vec: %s" % negRightVec)
                        negRightBias = newModel.predBias(negRightIndex)
                        print("neg right entity bias: %s" % negRightBias)

                        # find out the scores
                        crt = newModel.crtScore(leftVec, leftBias, relationIndex, rightVec, rightBias)
                        print("crt is %s" % crt)
                        crtln = newModel.crtScore(negLeftVec, negLeftBias, relationIndex, rightVec, rightBias)
                        print("crtln is %s" % crtln)
                        crtrn = newModel.crtScore(leftVec, leftBias, relationIndex, negRightVec, negRightBias)
                        print("crtrn is %s" % crtrn)

                        costl = newModel.margincost(crt, crtln, 1)
                        print("costl is %s" % costl)
                        costr = newModel.margincost(crt, crtrn, 1)
                        print("costr is %s" % costr)

                        exit()


            # if epoch_count % 10 == 9:

            # print out every epoch
            print("Finished epoch %s" % (epoch_count + 1))
            print("Loss is %s" % epochloss)

            # TODO: shall we do the normalisation? there are different choices
            # normalise entity embeddings at the end of an epoch, but don't normalise relations
            # TransE
            # transE.entityEmbedding.weight.data = F.normalize(transE.entityEmbedding.weight.data).detach()
            # NewModel
            # try not to normalise for now: entity vec will become nan
            newModel.predVec.weight.data = F.normalize(newModel.predVec.weight.data).detach()
            newModel.relationEmbedding.weight.data = F.normalize(newModel.predVec.weight.data).detach()


            # finalEpochLoss = epochloss


        # model evaluation using ranking function after 50 epochs

        # print("Training finishes after 1 epoch")
        # print("The loss for the last epoch is %s" % finalEpochLoss)
        #
        # print("Evaluation:")
        # print("Total number of entities is %s" % newModel.predVec.num_embeddings)
        # rankValid = newModel.rankScore(self.validLeftIndex, self.validRelIndex, self.validRightIndex)
        # meanRankValid = np.mean(rankValid[0] + rankValid[1])
        # print("Mean rank for valid is %s" % meanRankValid)
        #
        # rankTest = newModel.rankScore(self.testLeftIndex, self.testRelIndex, self.testRightIndex)
        # meanRankTest = np.mean(rankTest[0] + rankTest[1])
        # print("Mean rank for test is %s" % meanRankTest)


# main program
trainingInstance = train()
trainingInstance.train(0.01, 1)

# hyper parameter tuning starts here

# simi_functions = ["L1", "L2", "Dot"]
# learning_rates = [0.01, 0.1]
# margins = [1, 2]
#
# for simi in simi_functions:
#     for lr in learning_rates:
#         for margin in margins:
#             trainingInstance.train(simi, lr, margin)
#             print("Let's rest for 10 seconds")
#             print("\n")
#             time.sleep(10)




