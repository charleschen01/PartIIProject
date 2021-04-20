import torch

from RelationEmbeddingModel import RelationEmbeddingModel

class TransE(RelationEmbeddingModel):


    def __init__(self, simi, **kwds):

        # **kwds refers to the parameters in the base model
        super().__init__(**kwds)

        self.whoami = "TransE"
        self.simi = simi

    def tripletScore(self, leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):
        if self.simi == "L1":
            return -torch.sum(torch.abs(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1)
        elif self.simi == "L2":
            return -torch.sqrt(torch.sum(torch.square(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1))
        else:
            # TODO: throw an error here that we passed in an invalid simi argument
            pass







