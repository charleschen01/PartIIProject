import numpy as np
import pickle


def validSetCheck(trainLeftIndex, trainRightIndex, validLeftIndex, validRightIndex):
    # check if synsets in validation set all have appeared in training set
    trainSynsets = set(np.concatenate((trainLeftIndex, trainRightIndex)))
    validSynsets = set(np.concatenate((validLeftIndex, validRightIndex)))
    print("Total number of synsets in validation set is %s" % len(validSynsets))
    print("Total number of triplets in validation set is %s" % len(validLeftIndex))
    validNewSynsets = validSynsets.difference(trainSynsets)  # we want to obtain validSynsets - trainSynsets
    print("Number of new synsets in validation set is %s" % len(validNewSynsets))

    newSynsetsTripletIndices = [i for i in range(len(validLeftIndex)) if validLeftIndex[i] in validNewSynsets
                                or validRightIndex[i] in validNewSynsets]
    print("Number of affected triplets is %s" % len(newSynsetsTripletIndices))

def extractHypernym(trainLeftIndex, trainRelIndex, trainRightIndex,
                    validLeftIndex, validRelIndex, validRightIndex):
    # we find out the indices corresponding to Hypernym in training set
    print("Total number of triplets in training set is %s" % len(trainRelIndex))
    trainHypernymIndices = np.where((trainRelIndex == 3) | (trainRelIndex == 4))[0]
    print("Total number of Hypernym triplets is %s" % len(trainHypernymIndices))
    print()

    # we find out the indices corresponding to Hypernym in validation set
    print("Total number of triplets in validation set is %s" % len(validRelIndex))
    validHypernymIndices = np.where((validRelIndex == 3) | (validRelIndex == 4))[0]
    print("Total number of Hypernym triplets is %s" % len(validHypernymIndices))

    # we filter out triplets in validation whose synsets never appear in training, for a fairer comparison
    trainHypernymSynsets = set(np.concatenate((trainLeftIndex[trainHypernymIndices], trainRightIndex[trainHypernymIndices])))
    filteredValidHypernymIndices = [i for i in validHypernymIndices
                                    if validLeftIndex[i] in trainHypernymSynsets
                                    and validRightIndex[i] in trainHypernymSynsets]
    print("Total number of Hypernym triplets in filtered validation set is %s" % len(filteredValidHypernymIndices))

    # return a tuple for all the indices
    return (trainLeftIndex[trainHypernymIndices],
            trainRelIndex[trainHypernymIndices],
            trainRightIndex[trainHypernymIndices],
            validLeftIndex[filteredValidHypernymIndices],
            validRelIndex[filteredValidHypernymIndices],
            validRightIndex[filteredValidHypernymIndices])

def main():

    from train import convert2idx, load_file

    # TODO: we read wn18rr training and validation set, obtain hypernym only triplets, and perform filtering on validation triplets based on appearance on training set

    trainLeft = load_file("./data/wn18rr_train_lhs")
    trainRel = load_file("./data/wn18rr_train_rel")
    trainRight = load_file("./data/wn18rr_train_rhs")

    # validation set
    validLeft = load_file("./data/wn18rr_valid_lhs")
    validRel = load_file("./data/wn18rr_valid_rel")
    validRight = load_file("./data/wn18rr_valid_rhs")

    # index conversion
    # these are the indices that we can directly feed into our first layer
    trainLeftIndex = convert2idx(trainLeft)
    trainRelIndex = convert2idx(trainRel)
    trainRightIndex = convert2idx(trainRight)
    validLeftIndex = convert2idx(validLeft)
    validRelIndex = convert2idx(validRel)
    validRightIndex = convert2idx(validRight)

    extractHypernym(trainLeftIndex, trainRelIndex, trainRightIndex,
                    validLeftIndex, validRelIndex, validRightIndex)

if __name__ == "__main__":
    main()






