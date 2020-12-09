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



def load_file(path):
    return scipy.sparse.csr_matrix(pickle.load(open(path, 'rb')), dtype='float32')


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]
    # return the indices of the non-zero elements, arranged in the order of triplets
    # this is actually a really smart way to compress the sparse matrix


class trainTransE:

    def __init__(self):

        # load the 4 dictionaries for entities and relations
        entity2id = pickle.load(open("./data/wn18_subset_entity2id", 'rb'))
        id2entity = pickle.load(open("./data/wn18_subset_id2entity", 'rb'))
        id2relation = pickle.load(open("./data/wn18_subset_id2relation", 'rb'))
        relation2id = pickle.load(open("./data/wn18_subset_relation2id", 'rb'))

        self.numEntity = len(entity2id)
        self.numRelation = len(relation2id)
        self.total_epoch = 10   # the original figure is 1000, 10 is just for testing purpose
        self.nbatches = 100


    # we need to obtain indices for left entity, right entity, relation, as well as negative left and right entities
    def train(self):

        transE = TransE(self.numEntity, self.numRelation)

        print(transE.parameters())

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

        # index conversion
        # these are the indices that we can directly feed into our first layer
        trainLeftIndex = convert2idx(trainLeft)
        trainRightIndex = convert2idx(trainRight)
        trainRelIndex = convert2idx(trainRel)
        validLeftIndex = convert2idx(validLeft)
        validRightIndex = convert2idx(validRight)
        validRelIndex = convert2idx(validRel)
        testLeftIndex = convert2idx(testLeft)
        testRightIndex = convert2idx(testRight)
        testRelIndex = convert2idx(testRel)

        # start training process
        inputsize = len(trainLeftIndex)
        batchsize = inputsize//self.nbatches

        optimizer = optim.Adam(transE.parameters(), lr=0.01)

        print("Start training...")

        for epoch_count in range(self.total_epoch):

            print("Training epoch %s" % epoch_count)

            epochloss = 0

            # shuffling
            order = np.random.permutation(inputsize)
            trainLeftIndex = trainLeftIndex[order]
            trainRightIndex = trainRightIndex[order]
            trainRelIndex = trainRelIndex[order]

            # negatives
            # we choose a random index
            trainNegLeftIndex = random.sample(range(inputsize), inputsize)
            trainNegRightIndex = random.sample(range(inputsize), inputsize)

            for i in range(self.nbatches):
                batchLeftIndex = torch.LongTensor(trainLeftIndex[i*batchsize:(i+1)*batchsize])
                batchRightIndex = torch.LongTensor(trainRightIndex[i*batchsize:(i+1)*batchsize])
                batchRelIndex = torch.LongTensor(trainRelIndex[i*batchsize:(i+1)*batchsize])
                batchNegLeftIndex = torch.LongTensor(trainNegLeftIndex[i*batchsize:(i+1)*batchsize])
                batchNegRightIndex = torch.LongTensor(trainNegRightIndex[i*batchsize:(i+1)*batchsize])

                x = (batchLeftIndex, batchRightIndex, batchRelIndex, batchNegLeftIndex, batchNegRightIndex)
                loss = transE(x)
                loss.backward()
                optimizer.step()
                epochloss += loss

            print("Finish epoch %s" % epoch_count)
            print("Epoch loss is %s" % epochloss)


trainingInstance = trainTransE()
trainingInstance.train()





# feed embeddings into neural network batch by batch
# two layers of NN:
# first layer converts a high-dimensional vector (sparse matrix) to low-dimensional vector (embeddings)
# (need to figure our how to do this, probably incorporate the embedding here)

# write a validation function which allows us to stop early