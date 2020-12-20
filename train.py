# similar to WN_parse.py
# convert triplets to sparse matrices which can serve as inputs to the neural network
import scipy.sparse
import pickle
import numpy as np
from transE import TransE
import random

# optimiser
import torch
import torch.optim as optim

import torch.nn.functional as F

import time



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


class trainTransE:

    def __init__(self):

        # load the 4 dictionaries for entities and relations

        # subset
        entity2id = pickle.load(open("./data/wn18_subset_entity2id", 'rb'))
        id2entity = pickle.load(open("./data/wn18_subset_id2entity", 'rb'))
        id2relation = pickle.load(open("./data/wn18_subset_id2relation", 'rb'))
        relation2id = pickle.load(open("./data/wn18_subset_relation2id", 'rb'))

        # full dataset
        # entity2id = pickle.load(open("./data/wn18_entity2id", 'rb'))
        # id2entity = pickle.load(open("./data/wn18_id2entity", 'rb'))
        # id2relation = pickle.load(open("./data/wn18_id2relation", 'rb'))
        # relation2id = pickle.load(open("./data/wn18_relation2id", 'rb'))

        self.numEntity = len(entity2id)
        self.numRelation = len(relation2id)
        # the original figure is 1000, 10 is just for testing purpose
        # TODO: need to adjust it for the actual training
        self.total_epoch = 50
        # divide our triplets into 100 batches
        self.nbatches = 100

        # we need to obtain indices for left entity, right entity, relation, as well as negative left and right entities
        # which we can then feed into the network

        # the format of datasets below is parse csr files of size (no. of entities/relations * no. of triplets)

        # use subset

        # positives
        trainLeft = load_file("./data/wn18_subset_train_lhs")
        trainRight = load_file("./data/wn18_subset_train_rhs")
        trainRel = load_file("./data/wn18_subset_train_rel")

        # validation set
        validLeft = load_file("./data/wn18_subset_valid_lhs")
        validRight = load_file("./data/wn18_subset_valid_rhs")
        validRel = load_file("./data/wn18_subset_valid_rel")

        # test set
        testLeft = load_file("./data/wn18_subset_test_lhs")
        testRight = load_file("./data/wn18_subset_test_rhs")
        testRel = load_file("./data/wn18_subset_test_rel")

        # use full dataset

        # trainLeft = load_file("./data/wn18_train_lhs")
        # trainRight = load_file("./data/wn18_train_rhs")
        # trainRel = load_file("./data/wn18_train_rel")
        #
        # # validation set
        # validLeft = load_file("./data/wn18_valid_lhs")
        # validRight = load_file("./data/wn18_valid_rhs")
        # validRel = load_file("./data/wn18_valid_rel")
        #
        # # test set
        # testLeft = load_file("./data/wn18_test_lhs")
        # testRight = load_file("./data/wn18_test_rhs")
        # testRel = load_file("./data/wn18_test_rel")

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

    def train(self, simi, lr, margin):

        # initialise the neural network
        transE = TransE(self.numEntity, self.numRelation, simi, margin)



        # start training process
        inputsize = len(self.trainLeftIndex)
        print("Input size is %s (number of triplets)" % inputsize)
        batchsize = inputsize//self.nbatches

        optimizer = optim.Adam(transE.parameters(), lr)

        print("Start training...")
        print("Parameters: Similarity Function: {0}, Learning rate: {1}, Margin: {2}".format(simi, lr, margin))

        finalEpochLoss = 0

        for epoch_count in range(self.total_epoch):

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
                batchLeftIndex = torch.LongTensor(trainLeftIndex[i*batchsize:(i+1)*batchsize])
                batchRightIndex = torch.LongTensor(trainRightIndex[i*batchsize:(i+1)*batchsize])
                batchRelIndex = torch.LongTensor(trainRelIndex[i*batchsize:(i+1)*batchsize])
                batchNegLeftIndex = torch.LongTensor(trainNegLeftIndex[i*batchsize:(i+1)*batchsize])
                batchNegRightIndex = torch.LongTensor(trainNegRightIndex[i*batchsize:(i+1)*batchsize])

                x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex)
                transE.zero_grad()
                loss = transE(x)  # dynamically creating the computation graph
                loss.backward()
                optimizer.step()

                epochloss += loss

            if epoch_count % 10 == 9:
                print("Finished epoch %s" % (epoch_count + 1))

            # normalise entity embeddings at the end of an epoch
            transE.entityEmbedding.weight.data = F.normalize(transE.entityEmbedding.weight.data).detach()

            finalEpochLoss = epochloss


        # model evaluation using ranking function after 50 epochs

        print("Training finishes after 50 epochs")
        print("The loss for the last epoch is %s" % finalEpochLoss)

        print("Evaluation:")
        print("Total number of entities is %s" % transE.entityEmbedding.num_embeddings)
        rankValid = transE.rankScore(self.validLeftIndex, self.validRelIndex, self.validRightIndex)
        meanRankValid = np.mean(rankValid[0] + rankValid[1])
        print("Mean rank for valid is %s" % meanRankValid)

        rankTest = transE.rankScore(self.testLeftIndex, self.testRelIndex, self.testRightIndex)
        meanRankTest = np.mean(rankTest[0] + rankTest[1])
        print("Mean rank for test is %s" % meanRankTest)


# hyper parameter tuning starts here

trainingInstance = trainTransE()

simi_functions = ["L1", "L2", "Dot"]
learning_rates = [0.01, 0.1]
margins = [1, 2]

for simi in simi_functions:
    for lr in learning_rates:
        for margin in margins:
            trainingInstance.train(simi, lr, margin)
            print("Let's rest for 10 seconds")
            print("\n")
            time.sleep(10)




