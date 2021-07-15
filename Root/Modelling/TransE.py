import torch

from Root.Modelling.RelationEmbeddingModel import RelationEmbeddingModel

class TransE(RelationEmbeddingModel):


    def __init__(self, norm, **kwds):

        # **kwds refers to the parameters in the base model
        super().__init__(**kwds)

        self.whoami = "TransE"
        self.norm = norm

    def tripletScore(self, leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):
        if self.norm == "L1":
            return -torch.sum(torch.abs(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1)
        elif self.norm == "L2":
            return -torch.sqrt(torch.sum(torch.square(leftEnEmbeddings + relEmbeddings - rightEnEmbeddings), dim=1))
        else:
            raise ValueError("Invalid norm input!")