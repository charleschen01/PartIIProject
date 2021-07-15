import scipy.sparse
import pickle
import numpy as np
import time
from Root.Modelling.TransE import TransE
from Root.Modelling.DistMult import DistMult
from Root.Modelling.NewModel import NewModel

# optimiser
import torch
import torch.optim as optim

import torch.nn.functional as F

import math
from random import shuffle

class Experiment:

    # helper function to load sparse matrix
    def load_file(self, path):
        return scipy.sparse.csr_matrix(pickle.load(open(path, 'rb')), dtype='float32')

    # helper function to convert sparse matrix to a list of indices
    def convert2idx(self, spmat):

        # nonzero() returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix
        rows, cols = spmat.nonzero()
        return rows[np.argsort(cols)]
        # return the indices of the non-zero elements, arranged in the order of triplets
        # a smart way to compress the sparse matrix
        # makes sure different datasets will maintain the same order
        # eventually we will be able to get [idx for entity in triplet 1, idx for entity in triplet 2, ...]

    # dataset and GPU are Experiment specific parameters
    def __init__(self, dataset, GPU=False):

        # load the 4 dictionaries for entities and relations
        # and another 2 dictionary for generating corrupted triplets

        # specify whether it's wn18 or wn18rr
        self.dataset = dataset

        # load in the entity and relation datasets in the form of dictionary
        self.synset2id = pickle.load(open("./data/%s_entity2id" % self.dataset, 'rb'))
        self.id2synset = pickle.load(open("./data/%s_id2entity" % self.dataset, 'rb'))
        self.id2relation = pickle.load(open("./data/%s_id2relation" % self.dataset, 'rb'))
        self.relation2id = pickle.load(open("./data/%s_relation2id" % self.dataset, 'rb'))

        # load in {(left, rel) -> right} mapping and {(rel, right) -> left} mapping
        self.leftRel2Right = pickle.load(open("./data/%s_leftRel2Right" % self.dataset, 'rb'))
        self.relRight2Left = pickle.load(open("./data/%s_relRight2Left" % self.dataset, 'rb'))

        self.numSynset = len(self.synset2id)
        self.numRelation = len(self.relation2id)  # should be 18 for wn18 and 11 for wn18rr

        self.GPU = GPU

        # we need to obtain indices for left entity, right entity, relation, as well as negative left and right entities
        # which we can then feed into the network

        # load in the sparse csr files
        # the format of datasets below is sparse csr files of size (no. of entities/relations * no. of triplets)

        trainLeft = self.load_file("./data/%s_train_lhs" % self.dataset)
        trainRel = self.load_file("./data/%s_train_rel" % self.dataset)
        trainRight = self.load_file("./data/%s_train_rhs" % self.dataset)


        # validation set
        validLeft = self.load_file("./data/%s_valid_lhs" % self.dataset)
        validRel = self.load_file("./data/%s_valid_rel" % self.dataset)
        validRight = self.load_file("./data/%s_valid_rhs" % self.dataset)


        # test set
        testLeft = self.load_file("./data/%s_test_lhs" % self.dataset)
        testRel = self.load_file("./data/%s_test_rel" % self.dataset)
        testRight = self.load_file("./data/%s_test_rhs" % self.dataset)

        # index conversion
        # these are lists of indices that we can directly feed into our first layer
        self.trainLeftIndex = self.convert2idx(trainLeft)
        self.trainRelIndex = self.convert2idx(trainRel)
        self.trainRightIndex = self.convert2idx(trainRight)
        self.validLeftIndex = self.convert2idx(validLeft)
        self.validRelIndex = self.convert2idx(validRel)
        self.validRightIndex = self.convert2idx(validRight)
        self.testLeftIndex = self.convert2idx(testLeft)
        self.testRelIndex = self.convert2idx(testRel)
        self.testRightIndex = self.convert2idx(testRight)

    # in this function, we specify the model the hyper-parameters we want to use
    # norm is only passed when we use TransE or NewModel
    def trainModel(self, model, optimizer, lr, margin, dimension, norm="None",
                   num_epochs = 50,
                   num_batches = 10):

        # initialise the neural network
        if model == "TransE":
            self.model = TransE(norm=norm,
                                numEntity=self.numSynset,
                                numRelation=self.numRelation,
                                dimension=dimension,
                                margin=margin,
                                dataset=self.dataset,
                                GPU=self.GPU)
        elif model == "DistMult":
            self.model = DistMult(numEntity=self.numSynset,
                                  numRelation=self.numRelation,
                                  dimension=dimension,
                                  margin=margin,
                                  dataset=self.dataset,
                                  GPU=self.GPU)
        elif model == "NewModel":
            self.model = NewModel(norm=norm,
                                  numEntity=self.numSynset,
                                  numRelation=self.numRelation,
                                  dimension=dimension,
                                  margin=margin,
                                  dataset=self.dataset,
                                  GPU=self.GPU)

        # perform training on GPU if we are on GPU mode
        if self.GPU:
            self.model = self.model.to(torch.device('cuda:0'))

        # here, we start the training process
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr)
        elif optimizer == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr)
        else:
            raise ValueError("Invalid optimizer input!")

        print("Start training on {0}...")
        print("Parameters: "
              "Optimizer: {1}, "
              "Learning rate: {2}, "
              "Margin: {3}, "
              "Dimension: {4}, "
              "Norm: {5},"
              "Number of epochs: {6}, "
              "Number of batches: {7}".format(model, optimizer, lr, margin, dimension, norm, num_epochs, num_batches))

        # if we are using the new model, we want to call a seperate training function, because we need to differentiate
        # different relations
        if self.model.whoami == "NewModel":
            self.train_newModel(num_epochs, num_batches)
            return

        # start training process
        inputSize = len(self.trainLeftIndex) # this is the total size of the triplets
        print("Input size is %s (number of triplets)" % inputSize)
        batchSize = math.ceil(inputSize/num_batches)

        finalEpochLoss = 0

        for epoch_count in range(num_epochs):

            # print("Start epoch %s" % epoch_count)
            epochLoss = 0

            # shuffle the indices
            order = np.random.permutation(inputSize)
            trainLeftIndex = self.trainLeftIndex[order]
            trainRightIndex = self.trainRightIndex[order]
            trainRelIndex = self.trainRelIndex[order]

            # negatives
            # we choose a random index
            trainNegLeftIndex = np.random.choice(self.numSynset, inputSize, replace=True)
            trainNegRightIndex = np.random.choice(self.numSynset, inputSize, replace=True)

            # we need to make sure that trainLeftIndex and trainNegRightIndex do not have any term that are the same
            # similarly, trainRightIndex and trainNegLeftIndex should not have any same term
            # TODO: why do we need to do so? is it because errors will be thrown otherwise?
            # TODO: I cannot think of a good reason; this might be necessary only for the new model
            # I disable it for now, should be fine

            # for bad_index in np.where(trainLeftIndex - trainNegRightIndex == 0)[0]:
            #     while trainNegRightIndex[bad_index] == trainLeftIndex[bad_index]:
            #         # keep choosing random values until not the same
            #         trainNegRightIndex[bad_index] = np.random.choice(self.numSynset)
            #
            # for bad_index in np.where(trainRightIndex - trainNegLeftIndex == 0)[0]:
            #     while trainRightIndex[bad_index] == trainNegLeftIndex[bad_index]:
            #         # keep choosing random values until not the same
            #         trainNegLeftIndex[bad_index] = np.random.choice(self.numSynset)

            for i in range(num_batches):

                start = i*batchSize
                # the last batch will probably not have the full batch size, so we end at inputSize
                end = min((i+1) * batchSize, inputSize)

                # we use LongTensor because our embeddings expect long tensor inputs
                batchLeftIndex = torch.LongTensor(trainLeftIndex[start:end])
                batchRightIndex = torch.LongTensor(trainRightIndex[start:end])
                batchRelIndex = torch.LongTensor(trainRelIndex[start:end])
                batchNegLeftIndex = torch.LongTensor(trainNegLeftIndex[start:end])
                batchNegRightIndex = torch.LongTensor(trainNegRightIndex[start:end])

                x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex)
                self.model.zero_grad()
                loss = self.model(x)  # dynamically creating the computation graph
                loss.backward()
                self.optimizer.step()

                epochLoss += loss

            # print out every epoch
            # print("Finished epoch %s" % (epoch_count))
            # print("Loss is %s" % epochloss)

            # normalise entity embeddings at the end of an epoch, but don't normalise relations
            self.model.entityEmbedding.weight.data = F.normalize(self.model.entityEmbedding.weight.data).detach()
            finalEpochLoss = epochLoss

        print("Training finishes after %s epoch" % num_epochs)
        print("The loss for the last epoch is %s" % finalEpochLoss)

        # training finishes, we move the model back to CPU
        if self.GPU:
            self.model = self.model.to(torch.device('cpu'))

    def train_newModel(self, num_epochs, num_batches):

        # if we are using new model, we want to group triplets according to relation category here

        # assign the triplets into groups according to the relation type to speed up the training
        # we have four different groups
        # each is a list of index numpy array for that particular relation
        # [[...],[...],[...],[...]]
        trainLeftIndexGrouped = [None for _ in range(4)]
        trainRightIndexGrouped = [None for _ in range(4)]
        trainRelIndexGrouped = [None for _ in range(4)]
        trainNegLeftIndexGrouped = [None for _ in range(4)]
        trainNegRightIndexGrouped = [None for _ in range(4)]

        # create an array showing relation group number for each specific triplet
        # groupAllocation will be a list of group numbers
        groupAllocation = np.array([self.model.relGroup(i) for i in self.trainRelIndex])

        # now we use the group allocation to categorise indices into different groups
        for group in range(4):
            indicesForThisGroup = np.where(groupAllocation == group)[0]
            trainLeftIndexGrouped[group] = self.trainLeftIndex[indicesForThisGroup]
            trainRightIndexGrouped[group] = self.trainRightIndex[indicesForThisGroup]
            # TODO: we need to change this if we are using DistMult for synonym case
            if group == 3 or group == 2:
                # we only need relation embedding for context shift case and synonym (if distmult)
                # when group is 0 or 1 (or 2), rel index array will be None
                trainRelIndexGrouped[group] = self.trainRelIndex[indicesForThisGroup]

        # we've done our grouping, now start training process
        inputSize = len(self.trainLeftIndex) # this is the total number of triplets
        print("Input size is %s (number of triplets)" % inputSize)
        batchSize = math.ceil(inputSize/num_batches)
        # if we are using the new model, num_batches is only a reference and the actual number may be slightly bigger

        finalEpochLoss = 0

        for epoch_count in range(num_epochs):

            epochLoss = 0

            # for each relation group, we permutate indices randomly and also generate negative synsets
            for group in range(4):
                # if group contains no relation, size is 0, we skip
                size = len(trainLeftIndexGrouped[group])

                # the case of size 0 will happen for wn18rr
                # we simply skip this relation
                if size == 0:
                    continue

                # create a random order
                order = np.random.permutation(size)
                trainLeftIndexGrouped[group] = trainLeftIndexGrouped[group][order]
                trainRightIndexGrouped[group] = trainRightIndexGrouped[group][order]
                # TODO: change here
                if group == 3 or group == 2:
                    # only use relation embedding when we are in the context shift case (or the synonym case)
                    trainRelIndexGrouped[group] = trainRelIndexGrouped[group][order]
                # generate negative synsets
                trainNegLeftIndexGrouped[group] = np.random.choice(self.numSynset, size, replace=True)
                trainNegRightIndexGrouped[group] = np.random.choice(self.numSynset, size, replace=True)

                # eliminate repetition of synset
                # having the same entity on both sides will throw us errors during training
                for bad_index in np.where(trainLeftIndexGrouped[group] - trainNegRightIndexGrouped[group] == 0)[0]:
                    while trainNegRightIndexGrouped[group][bad_index] == trainLeftIndexGrouped[group][bad_index]:
                        # keep choosing random values until not the same
                        trainNegRightIndexGrouped[group][bad_index] = np.random.choice(self.numSynset)

                for bad_index in np.where(trainRightIndexGrouped[group] - trainNegLeftIndexGrouped[group] == 0)[0]:
                    while trainRightIndexGrouped[group][bad_index] == trainNegLeftIndexGrouped[group][bad_index]:
                        # keep choosing random values until not the same
                        trainNegLeftIndexGrouped[group][bad_index] = np.random.choice(self.numSynset)

            # we loop over the four groups to put them into a whole list of batches

            # and then mix and match different groups
            # put all the groups into a list then permutate

            # create a list of tensor tuples (batchLeft, batchRight, batchRel, batchNegLeft, batchNegRight)
            batchList = []

            # create batches and put them into our group
            for group in range(4):
                size = len(trainLeftIndexGrouped[group]) # number of triplets for that relation

                if size == 0:
                    continue

                # here because we take the ceiling, we might end up having a higher number of batches
                for i in range(math.ceil(size/batchSize)):
                    start = i * batchSize
                    # the last batch probably will not have the full batchsize
                    end = min((i+1) * batchSize, size)
                    # we use LongTensor because our embeddings expect LongTensor inputs
                    batchLeftIndex = torch.LongTensor(trainLeftIndexGrouped[group][start:end])
                    batchRightIndex = torch.LongTensor(trainRightIndexGrouped[group][start:end])
                    # we only generate rel embeddings when group is 3
                    # TODO: need to include group 2 after we use distmult for synonynm
                    batchRelIndex = torch.LongTensor(trainRelIndexGrouped[group][start:end]
                                                     if group == 3 or group == 2
                                                     else [])
                    batchNegLeftIndex = torch.LongTensor(trainNegLeftIndexGrouped[group][start:end])
                    batchNegRightIndex = torch.LongTensor(trainNegRightIndexGrouped[group][start:end])

                    x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex, group)
                    batchList.append(x)

            # now we mismatch batches so that we don't train exclusively on one relation category before jumping on to
            # next
            # randomly permutate the batchList
            shuffle(batchList)

            # loop through the batch list to perform training
            for x in batchList:
                self.model.zero_grad()
                loss = self.model(x)  # dynamically creating the computation graph
                loss.backward()
                self.optimizer.step()
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

            # print out every epoch to make sure our model is working correctly
            # print("Batch count is %s" % batchCount)
            # print("Finished epoch %s" % (epoch_count))
            # print("Loss is %s" % epochLoss)

            # we normalise the predicate vector
            self.model.predVec.weight.data = F.normalize(self.model.predVec.weight.data).detach()
            finalEpochLoss = epochLoss

        print("Training finishes after %s epoch" % num_epochs)
        print("The loss for the last epoch is %s" % finalEpochLoss)

        if self.GPU:
            self.model = self.model.to(torch.device('cpu'))

    def evaluate(self, targetSet, relationSplit=False):

        # input:
        # targetSet is either "validation" or "test" or "training"
        # relationSplit is an additonal parameter to indicate whether we want to split relations

        # move to GPU, because evaluation can be parallelised
        if self.GPU:
            self.model = self.model.to(torch.device('cuda:0'))

        print("Evaluation on %s set:" % targetSet)
        print("Total number of synset is %s" % self.numSynset)

        # rank is in the form [leftRankList,leftReciRankList,rightRankList,rightReciRankList]
        rank = []
        relations = []
        if targetSet == "valid":
            rank = self.generateRank(self.validLeftIndex, self.validRelIndex, self.validRightIndex)
            if relationSplit:
                relations = self.validRelIndex # relations used to classify different relations
        elif targetSet == "test":
            rank = self.generateRank(self.testLeftIndex, self.testRelIndex, self.testRightIndex)
            if relationSplit:
                relations = self.testRelIndex
        elif targetSet == "train":
            # we evaluate the first 2000 triplets of training set
            # maybe can increase this number after we can run our code on GPU
            rank = self.generateRank(self.trainLeftIndex[0:2000], self.trainRelIndex[0:2000], self.trainRightIndex[0:2000])
            if relationSplit:
                relations = self.trainRelIndex[0:2000]
        else:
            raise ValueError("Invalid target set argument!")

        # the overall dataset
        rankArray = np.asarray(rank[0] + rank[2])
        reciRankArray = np.asarray(rank[1] + rank[3])
        print("Overall results:")
        self.displayEvaluationResults(rankArray,reciRankArray)

        # include an additional evaluation metrics for different relations
        # here we analyse individual relation categories
        if relationSplit:
            groupArray = np.array([self.model.relGroup(rel) for rel in relations])
            for relGroup in range(4):
                # for each relation, we first get all the indices for that relation
                indices = np.where(groupArray == relGroup)[0]
                # then extracts out ranks for the correct relation
                rankArray = np.concatenate((np.asarray(rank[0])[indices], np.asarray(rank[2])[indices]))
                reciRankArray = np.concatenate((np.asarray(rank[1])[indices], np.asarray(rank[3])[indices]))
                print("Results for relation %s:" % relGroup)
                self.displayEvaluationResults(rankArray, reciRankArray)

        # switch back to CPU
        if self.GPU:
            self.model = self.model.to(torch.device('cpu'))

    # a helper function to show evaluation results
    def displayEvaluationResults(self, rankArray, reciRankArray):

        total = rankArray.size
        # check if rankArray is empty
        # this will happen for the hyponym case when we use WN18RR dataset
        if total == 0:
            return
        meanRank = np.mean(rankArray)
        print("Mean rank is %s" % meanRank)
        meanReciRank = np.mean(reciRankArray)
        print("Mean reciprocal rank is %s" % meanReciRank)
        hitsAt10 = np.where(rankArray <= 10)[0].size / total * 100
        hitsAt3 = np.where(rankArray <= 3)[0].size / total * 100
        hitsAt1 = np.where(rankArray == 1)[0].size / total * 100
        print("Hits@10 is %s" % hitsAt10)
        print("Hits@3 is %s" % hitsAt3)
        print("Hits@1 is %s" % hitsAt1)
        print()

    def generateRank(self, idxLeft, idxRel, idxRight):

        # calculate predicted ranks for triplets

        # inputs:
        # idxLeft, idxRight and idxRel are all lists of indices
        # outputs:
        # a list [leftRankList,leftReciRankList,rightRankList,rightReciRankList]

        leftRankList = []
        leftReciRankList = []
        rightRankList = []
        rightReciRankList = []

        for left, rel, right in zip(idxLeft, idxRel, idxRight):

            # calculate rank of left
            # we need to calculate the score of all 'left' entities given rel and right

            # here we make sure that leftCrtScores does not include correct entities, need to find out the left entities
            #  that would have result in correct triplet and remove it from the list of left entities; however, we do
            #  need to include the correct left entity for that particular triplet

            # find out the correct entities that we want to ignore; we don't not treat them as corrupted
            # but we don't ignore the left entity itself
            leftIgnoreList = [entity for entity in self.relRight2Left[(rel, right)] if entity != left]

            # if we are using the new model and the relation is synonym, we want to add right to the ignoreList
            # having the same entity on both sides might give the best score
            # we don't want to include a triplet that gives the best score
            # or, it doesn't make much sense to do scoring for a triplet where both sides are the same
            leftIgnoreList.append(right)

            # generate the scores for all possible left entities first
            leftScores = self.model.evaluateSubjectScores(rel, right)

            # our ranking favours higher scores
            # set scores for ignored entities to be -infinite, so that we can ignore them in our ranking
            # in this way we only get sensible scores for corrupted triplets
            leftScores[leftIgnoreList] = float('-inf')

            # get ranking of idx from best to worst
            # for this list,
            # items are idx:    [4, 6, 46, 932, 13, ...]
            # index is ranking: [0, 1, 2, 3, 4, ...]
            # we then want to get the position of a particular index
            # use the where operation in numpy
            leftIdxRank = torch.argsort(leftScores, descending=True)

            # switch back to CPU once we start dealing with numpy arrays
            if self.GPU:
                leftIdxRank = leftIdxRank.to(torch.device('cpu'))

            # here we try to find out the rank of our left predicate
            leftRank = np.where(leftIdxRank == left)[0] # leftRank comes out as a list of one element
            leftRankList.extend(leftRank+1)  # plus 1 because we want rank to start from 1
            leftReciRankList.append(1/(leftRank[0]+1))

            # generate the right corrupted triplets in a similar fashion
            rightIgnoreList = [entity for entity in self.leftRel2Right[(left, rel)] if entity != right]
            rightIgnoreList.append(left)

            # similarly, calculate rank of right
            rightScores = self.model.evaluateObjectScores(left, rel)

            rightScores[rightIgnoreList] = float('-inf')
            rightIdxRank = torch.argsort(rightScores, descending=True)

            # switch back to cpu once we start dealing with numpy arrays, this time for the ranks of the right side
            if self.GPU:
                rightIdxRank = rightIdxRank.to(torch.device('cpu'))

            rightRank = np.where(rightIdxRank == right)[0]
            rightRankList.extend(rightRank+1)
            rightReciRankList.append(1/(rightRank[0]+1))

        return [leftRankList, leftReciRankList, rightRankList, rightReciRankList]

def main():
    startTime = time.localtime()

    e = Experiment("wn18rr", GPU=False)
    e.trainModel(model="NewModel",
                 optimizer="Adagrad",
                 lr=0.1,
                 margin=1,
                 dimension=100,
                 norm="L2",
                 num_epochs=100,
                 num_batches=10)
    e.evaluate("valid", relationSplit=False)

    endTime = time.localtime()

    duration = (time.mktime(endTime) - time.mktime(startTime))/60
    print("Total duration is %s minutes" % duration)

    # transE
    # trainingInstance.train_transE(0.01, "L2", 1, 20)
    # trainingInstance.evaluate("valid")

    # dimensions = [20, 50]
    # simi_functions = ["L1", "L2", "Dot"]

    #
    # results = {}
    #
    # startTime = time.localtime()
    #
    # # load dataset once at the start
    # trainingInstance = train()
    #
    # for k in dimensions:
    #     for simi in simi_functions:
    #         for lr in learning_rates:
    #             for margin in margins:
    #
    #                 # training
    #                 # each time we define a new model field within train class object
    #                 trainingInstance.train_transE(lr, simi, margin, k)
    #                 trainingInstance.evaluate("valid")
    #
    #                 print("Let's rest for 15 seconds")
    #                 print()
    #                 time.sleep(15)
    #
    # print(results)
    #
    # endTime = time.localtime()
    #
    # duration = (time.mktime(endTime) - time.mktime(startTime))/60
    # print("Total duration is %s minutes" % duration)


if __name__ == "__main__":
    main()