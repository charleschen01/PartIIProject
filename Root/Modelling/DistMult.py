import torch

from Root.Modelling.RelationEmbeddingModel import RelationEmbeddingModel

class DistMult(RelationEmbeddingModel):

    def __init__(self, **kwds):

        # **kwds refers to the parameters in the base model
        super().__init__(**kwds)
        self.whoami = "DistMult"

    def tripletScore(self, leftEnEmbeddings, relEmbeddings, rightEnEmbeddings):
        matrixProduct = torch.sum(leftEnEmbeddings*relEmbeddings*rightEnEmbeddings, dim=1)
        return matrixProduct