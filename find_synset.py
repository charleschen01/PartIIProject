# script finding subject and object synsets for each relation type

import pickle

f = open("./wn18/wordnet-mlj12-train.txt", 'r')
data = f.readlines()
f.close()

entity2word = pickle.load(open("./data/entity2word", 'rb'))

count = 0
for triplet in data:
    lhs, rel, rhs = triplet[:-1].split('\t')
    if rel == "_hyponym":
        print(entity2word[int(lhs)],entity2word[int(rhs)])
        count += 1
    if count == 100:
        break
