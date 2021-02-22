# similar to WN_parse.py
# convert triplets to sparse matrices which can serve as inputs to the neural network
import scipy.sparse
import pickle
import numpy as np
from transE import TransE
from NewModel import NewModel

# optimiser
import torch
import torch.nn as nn
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

        # full dataset
        entity2id = pickle.load(open("./data/wn18_entity2id", 'rb'))
        id2entity = pickle.load(open("./data/wn18_id2entity", 'rb'))
        id2relation = pickle.load(open("./data/wn18_id2relation", 'rb'))
        relation2id = pickle.load(open("./data/wn18_relation2id", 'rb'))

        self.numSynset = len(entity2id)
        self.numRelation = len(relation2id)  # should be 18 here

        # the original figure is 1000, 50 is just for testing purpose
        # TODO: need to adjust it for the actual training
        self.total_epoch = 50
        # divide our triplets into 100 batches
        self.nbatches = 100

        # we need to obtain indices for left entity, right entity, relation, as well as negative left and right entities
        # which we can then feed into the network

        # the format of datasets below is sparse csr files of size (no. of entities/relations * no. of triplets)

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

    def train_transE(self, lr, simi, margin, k):

        # initialise the neural network
        self.model = TransE(self.numSynset, self.numRelation, simi, margin, k)

        # start training process
        inputSize = len(self.trainLeftIndex) # this is the total size
        print("Input size is %s (number of triplets)" % inputSize)
        batchSize = inputSize//self.nbatches

        optimizer = optim.Adam(self.model.parameters(), lr)

        print("Start training...")
        print("Parameters: Learning rate: {0}, Margin: {1}, Similarity function: {2}, Dimension: {3}".format(lr, margin, simi, k))

        finalEpochLoss = 0

        for epoch_count in range(self.total_epoch):

            # print("Start epoch %s" % epoch_count)

            epochloss = 0

            # shuffling

            # TransE
            order = np.random.permutation(inputSize)
            trainLeftIndex = self.trainLeftIndex[order]
            trainRightIndex = self.trainRightIndex[order]
            trainRelIndex = self.trainRelIndex[order]

            # negatives
            # we choose a random index
            trainNegLeftIndex = np.random.choice(self.numSynset, inputSize, replace=True)
            trainNegRightIndex = np.random.choice(self.numSynset, inputSize, replace=True)

            # we need to make sure that trainLeftIndex and rightIndex do not have any term that are the same
            # similarly, trainRightIndex and leftIndex do not have any same term
            for bad_index in np.where(trainLeftIndex - trainNegRightIndex == 0)[0]:
                while trainNegRightIndex[bad_index] == trainLeftIndex[bad_index]:
                    # keep choosing random values until not the same
                    trainNegRightIndex[bad_index] = np.random.choice(self.numSynset)

            for bad_index in np.where(trainRightIndex - trainNegLeftIndex == 0)[0]:
                while trainRightIndex[bad_index] == trainNegLeftIndex[bad_index]:
                    # keep choosing random values until not the same
                    trainNegLeftIndex[bad_index] = np.random.choice(self.numSynset)

            # TransE

            for i in range(self.nbatches):

                # TODO: maybe convert this manual process into a smarter PyTorch one using dataloader?
                # TODO: what about the data left behind that doesn't fit into one batchsize?

                # we use LongTensor because our embeddings expect long tensor inputs
                batchLeftIndex = torch.LongTensor(trainLeftIndex[i*batchSize:(i+1)*batchSize])
                batchRightIndex = torch.LongTensor(trainRightIndex[i*batchSize:(i+1)*batchSize])
                batchRelIndex = torch.LongTensor(trainRelIndex[i*batchSize:(i+1)*batchSize])
                batchNegLeftIndex = torch.LongTensor(trainNegLeftIndex[i*batchSize:(i+1)*batchSize])
                batchNegRightIndex = torch.LongTensor(trainNegRightIndex[i*batchSize:(i+1)*batchSize])

                x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex)
                self.model.zero_grad()
                loss = self.model(x)  # dynamically creating the computation graph
                loss.backward()
                optimizer.step()

                epochloss += loss

            # print out every epoch
            # print("Finished epoch %s" % (epoch_count))
            # print("Loss is %s" % epochloss)

            # TODO: shall we do the normalisation? there are different choices
            # normalise entity embeddings at the end of an epoch, but don't normalise relations
            self.model.entityEmbedding.weight.data = F.normalize(self.model.entityEmbedding.weight.data).detach()
            finalEpochLoss = epochloss

        print("Training finishes after %s epoch" % self.total_epoch)
        print("The loss for the last epoch is %s" % finalEpochLoss)


    def evaluate_valid(self):

        # model evaluation using ranking function after certain epochs

        print("Evaluation on validation set:")
        print("Total number of entities is %s" % self.numSynset)

        rankValid = self.model.rankScore(self.validLeftIndex, self.validRelIndex, self.validRightIndex)
        meanRankValid = np.mean(rankValid[0] + rankValid[1])
        print("Mean rank for valid is %s" % meanRankValid)
        hitsAt10 = sum(rank <= 10 for rank in (rankValid[0] + rankValid[1])) / (len(rankValid[0])*2) * 100
        print("Hits at 10 for valid is %s" % hitsAt10)


    def evaluate_test(self):

        print("Evaluation on test set:")
        print("Total number of entities is %s" % self.numSynset)
        rankTest = self.model.rankScore(self.testLeftIndex, self.testRelIndex, self.testRightIndex)
        meanRankTest = np.mean(rankTest[0] + rankTest[1])
        print("Mean rank for test is %s" % meanRankTest)
        hitsAt10 = sum(rank <= 10 for rank in (rankTest[0] + rankTest[1])) / (len(rankTest[0])*2) * 100
        print("Hits at 10 for test is %s percent" % hitsAt10)


    def train_newModel(self, lr, margin, simi, k):

        # initialise the neural network
        # newModel
        self.model = NewModel(self.numSynset, self.numRelation, margin, simi, k)

        # if we are using new model, we want to group triplets according to relation category here

        # assign the triplets into groups according to the relation type to speed up the training
        # we have four different groups
        trainLeftIndexGrouped = [[], [], [], []]
        trainRightIndexGrouped = [[], [], [], []]
        trainRelIndexGrouped = [[], [], [], []]
        trainNegLeftIndexGrouped = [[], [], [], []]
        trainNegRightIndexGrouped = [[], [], [], []]

        # create an array of group numbers 0-3
        groupArray = np.array([self.model.relGroup(i) for i in self.trainRelIndex])
        for group in range(4):
            indicesForThisGroup = np.where(groupArray == group)[0]
            trainLeftIndexGrouped[group] = self.trainLeftIndex[indicesForThisGroup]
            trainRightIndexGrouped[group] = self.trainRightIndex[indicesForThisGroup]
            if group == 3:
                # we only need relation embedding in the translation case
                trainRelIndexGrouped[group] = self.trainRelIndex[indicesForThisGroup]

        # start training process
        inputSize = len(self.trainLeftIndex) # this is the total size
        print("Input size is %s (number of triplets)" % inputSize)
        batchSize = inputSize//self.nbatches
        # if we are using the new model, nbatches is only a reference and the actual number may differ

        optimizer = optim.Adam(self.model.parameters(), lr)

        print("Start training...")
        print("Parameters: Learning rate: {0}, Margin: {1}, Similarity function: {2}, Dimension: {3}".format(lr, margin, simi, k))

        finalEpochLoss = 0

        for epoch_count in range(self.total_epoch):

            epochLoss = 0

            # randomise the inputs and generate negative predicates
            for group in range(4):
                size = len(trainLeftIndexGrouped[group])
                # create a random order
                order = np.random.permutation(size)
                trainLeftIndexGrouped[group] = trainLeftIndexGrouped[group][order]
                trainRightIndexGrouped[group] = trainRightIndexGrouped[group][order]
                if group == 3:
                    # only use relation embedding when we are in the translation case
                    trainRelIndexGrouped[group] = trainRelIndexGrouped[group][order]
                # generate negative predicates
                trainNegLeftIndexGrouped[group] = np.random.choice(self.numSynset, size, replace=True)
                trainNegRightIndexGrouped[group] = np.random.choice(self.numSynset, size, replace=True)
                # eliminate repetition of synset
                for bad_index in np.where(trainLeftIndexGrouped[group] - trainNegRightIndexGrouped[group] == 0)[0]:
                    while trainNegRightIndexGrouped[group][bad_index] == trainLeftIndexGrouped[group][bad_index]:
                        # keep choosing random values until not the same
                        trainNegRightIndexGrouped[group][bad_index] = np.random.choice(self.numSynset)

                for bad_index in np.where(trainRightIndexGrouped[group] - trainNegLeftIndexGrouped[group] == 0)[0]:
                    while trainRightIndexGrouped[group][bad_index] == trainNegLeftIndexGrouped[group][bad_index]:
                        # keep choosing random values until not the same
                        trainNegLeftIndexGrouped[group][bad_index] = np.random.choice(self.numSynset)


            # we loop over the four groups to do training separately
            batchCount = 0
            for group in range(4):
                size = len(trainLeftIndexGrouped[group])
                for i in range(math.ceil(size/batchSize)):
                    start = i * batchSize
                    # the last batch probably will not have the full batchsize
                    end = min((i+1) * batchSize, size)
                    # we use LongTensor because our embeddings expect LongTensor inputs
                    batchLeftIndex = torch.LongTensor(trainLeftIndexGrouped[group][start:end])
                    batchRightIndex = torch.LongTensor(trainRightIndexGrouped[group][start:end])
                    batchRelIndex = torch.LongTensor(trainRelIndexGrouped[group][start:end] if group == 3 else [])
                    batchNegLeftIndex = torch.LongTensor(trainNegLeftIndexGrouped[group][start:end])
                    batchNegRightIndex = torch.LongTensor(trainNegRightIndexGrouped[group][start:end])

                    # training
                    x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex, group)
                    self.model.zero_grad()
                    loss = self.model(x)  # dynamically creating the computation graph
                    loss.backward()
                    optimizer.step()

                    # print("Finished batch {0}, group is {1}".format(batchCount,group))
                    # print("Loss is %s" % loss)
                    batchCount += 1
                    epochLoss += loss.item()

            #     # DEBUG: check predicates to see if there's any nan value
            #     all_indices = torch.LongTensor([i for i in range(newModel.numEntity)])
            #     for j, x in enumerate(newModel.predVec(all_indices)):
            #         if math.isnan(x[0]):
            #             print("predicate %s has become nan!" % j)
            #             print(x)
            #             # okay, we've found a nan; now we want to see what operation makes the predicate become nan
            #             badBatchIndex = 0
            #             for batchIdx in range(1414):
            #                 if batchLeftIndex[batchIdx] == j or batchRightIndex[batchIdx] == j or batchNegLeftIndex[batchIdx] == j or batchNegRightIndex[batchIdx] == j:
            #                     badBatchIndex = batchIdx
            #                     break
            #             if badBatchIndex == 0:
            #                 print("nan value not found in this batch")
            #             # we've find the badBatchIndex
            #             # now let's find all the training data there
            #             leftIndex = batchLeftIndex[badBatchIndex]
            #             print("left entity index: %s" % leftIndex)
            #             leftVec = startPredVec(leftIndex)
            #             print("left entity vec: %s" % leftVec)
            #             print("left entity new vec: %s" % newModel.predVec(leftIndex))
            #             leftBias = startPredBias(leftIndex)
            #             print("left entity bias: %s" % leftBias)
            #             # debug: we don't care about relation
            #             relationIndex = batchRelIndex[badBatchIndex]
            #             print("relation index: %s" % relationIndex)
            #             print("relation embedding: %s" % startRelEmbedding(relationIndex))
            #             rightIndex = batchRightIndex[badBatchIndex]
            #             print("right entity index: %s" % rightIndex)
            #             rightVec = startPredVec(rightIndex)
            #             print("right entity vec: %s" % rightVec)
            #             rightBias = startPredBias(rightIndex)
            #             print("right entity bias: %s" % rightBias)
            #             negLeftIndex = batchNegLeftIndex[badBatchIndex]
            #             print("negative left entity index: %s" % negLeftIndex)
            #             negLeftVec = startPredVec(negLeftIndex)
            #             print("neg left entity vec: %s" % negLeftVec)
            #             negLeftBias = startPredBias(negLeftIndex)
            #             print("neg left entity bias: %s" % negLeftBias)
            #             negRightIndex = batchNegRightIndex[badBatchIndex]
            #             print("negative right entity index: %s" % negRightIndex)
            #             negRightVec = startPredVec(negRightIndex)
            #             print("neg right entity vec: %s" % negRightVec)
            #             negRightBias = startPredBias(negRightIndex)
            #             print("neg right entity bias: %s" % negRightBias)
            #
            #             # debug: the score should be based on the current vec and bias after training
            #
            #             # find out the scores
            #             crt = newModel.crtScore(leftVec, leftBias, relationIndex, rightVec, rightBias)
            #             print("crt is %s" % crt)
            #             crtln = newModel.crtScore(negLeftVec, negLeftBias, relationIndex, rightVec, rightBias)
            #             print("crtln is %s" % crtln)
            #             crtrn = newModel.crtScore(leftVec, leftBias, relationIndex, negRightVec, negRightBias)
            #             print("crtrn is %s" % crtrn)
            #
            #             costl = newModel.margincost(crt, crtln, 1)
            #             print("costl is %s" % costl)
            #             costr = newModel.margincost(crt, crtrn, 1)
            #             print("costr is %s" % costr)
            #
            #             exit()

            # print out every epoch
            # print("Finished epoch %s" % (epoch_count))
            # print("Loss is %s" % epochLoss)

            # TODO: shall we do the normalisation? there are different choices
            # NewModel
            # we only normalise the predicate for now
            self.model.predVec.weight.data = F.normalize(self.model.predVec.weight.data).detach()
            # newModel.relationEmbedding.weight.data = F.normalize(newModel.predVec.weight.data).detach()
            finalEpochLoss = epochLoss


        # model evaluation using ranking function after certain epochs

        print("Training finishes after %s epoch" % self.total_epoch)
        print("The loss for the last epoch is %s" % finalEpochLoss)


# testing
# trainingInstance = train()
# trainingInstance.train_newModel(0.01,1,"L1")
# trainingInstance.evaluate_valid()

# hyperparameter tuning starts here

# we probably want to save the model obtained in each case

dimensions = [20, 50]
simi_functions = ["L1", "L2", "Dot"]
learning_rates = [0.001, 0.01, 0.1]
margins = [1, 2]

results = {}

startTime = time.localtime()

# load dataset once at the start
trainingInstance = train()

for k in dimensions:
    for simi in simi_functions:
        for lr in learning_rates:
            for margin in margins:

                # training
                # each time we define a new model field within train class object
                trainingInstance.train_transE(lr, simi, margin, k)
                trainingInstance.evaluate_valid()

                print("Let's rest for 15 seconds")
                print()
                time.sleep(15)

print(results)

endTime = time.localtime()

duration = (time.mktime(endTime) - time.mktime(startTime))/60
print("Total duration is %s minutes" % duration)






