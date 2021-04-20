# create dataset files for wn18, wn18_subset (which is not useful) and wn18rr
import numpy as np
import pickle
import scipy.sparse as sp

from collections import defaultdict

# we first assign ids for relations/entities and create mappings
# this would be useful later for creating the sparse matrix

entitylist = []
relationlist = []

for datatype in ['train', 'valid', 'test']:
    # TODO: configure here
    # f = open("./wn18_subset/wn18_%s_subset.txt" % datatype, 'r')
    # f = open("./wn18/wordnet-mlj12-%s.txt" % datatype, 'r')
    f = open("./wn18rr/wn18rr_%s.txt" % datatype, 'r')
    data = f.readlines()
    f.close()

    # obtain a set of entities and relations
    for triplet in data:
        lhs, rel, rhs = triplet[:-1].split('\t')
        entitylist += [int(lhs), int(rhs)]
        relationlist += [rel]

entityset = np.sort(list(set(entitylist)))
relset = np.sort(list(set(relationlist)))

print("{} entities".format(len(entityset)))
print("{} relations".format(len(relset)))
print(relset)

# we now fill in the dictionaries mapping entity and relation to id

entity2id = {}   # here, entity is the number used in the original dataset
id2entity = {}
relation2id = {}
id2relation = {}

entity_id = 0
for i in entityset:
    entity2id[i] = entity_id
    id2entity[entity_id] = i
    entity_id += 1

relation_id = 0
for i in relset:
    relation2id[i] = relation_id
    id2relation[relation_id] = i
    relation_id += 1

# TODO: configure here
# f1 = open('./data/wn18_subset_entity2id', 'wb')
# f2 = open('./data/wn18_subset_id2entity', 'wb')
# f3 = open('./data/wn18_subset_relation2id', 'wb')
# f4 = open('./data/wn18_subset_id2relation', 'wb')
# f1 = open('./data/wn18_entity2id', 'wb')
# f2 = open('./data/wn18_id2entity', 'wb')
# f3 = open('./data/wn18_relation2id', 'wb')
# f4 = open('./data/wn18_id2relation', 'wb')
f1 = open('./data/wn18rr_entity2id', 'wb')
f2 = open('./data/wn18rr_id2entity', 'wb')
f3 = open('./data/wn18rr_relation2id', 'wb')
f4 = open('./data/wn18rr_id2relation', 'wb')
pickle.dump(entity2id, f1)
pickle.dump(id2entity, f2)
pickle.dump(relation2id, f3)
pickle.dump(id2relation, f4)
f1.close()
f2.close()
f3.close()
f4.close()

# now we convert datasets into sparse matrices

# we would also create two dictionaries mapping {left,rel} to {right} and {rel,right} to {left}
# this is for all three datasets so we do it outside the loop
leftRel2Right = defaultdict(list)
relRight2Left = defaultdict(list)

for datatype in ['train', 'valid', 'test']:
    # TODO: configure here
    # f = open("./wn18_subset/wn18_%s_subset.txt" % datatype, 'r')
    f = open("./wn18/wordnet-mlj12-%s.txt" % datatype, 'r')
    # f = open("./wn18rr/wn18rr_%s.txt" % datatype, 'r')
    data = f.readlines()
    f.close()

    # create sparse matrix, specifying the size; number of columns is the number of triplets
    # each column only has one active bit, indicating the entity/relation we use
    input_left = sp.lil_matrix((entity_id, len(data)), dtype='float32')
    input_right = sp.lil_matrix((entity_id, len(data)), dtype='float32')
    input_rel = sp.lil_matrix((relation_id, len(data)), dtype='float32')

    # fill the sparse matrices
    # j indicates the triplet number
    j = 0
    for triplet in data:
        lhs, rel, rhs = triplet[:-1].split('\t')
        # if lhs == rhs, we simply discard the data
        if lhs == rhs:
            continue
        input_left[entity2id[lhs], j] = 1
        input_right[entity2id[rhs], j] = 1
        input_rel[relation2id[rel], j] = 1
        leftRel2Right[(entity2id[lhs],relation2id[rel])].append(entity2id[rhs])
        relRight2Left[(relation2id[rel],entity2id[rhs])].append(entity2id[lhs])
        j += 1

    # TODO: configure here
    f = open('./data/wn18_%s_lhs' % datatype, 'wb')
    g = open('./data/wn18_%s_rhs' % datatype, 'wb')
    h = open('./data/wn18_%s_rel' % datatype, 'wb')
    pickle.dump(input_left.tocsr(), f)
    pickle.dump(input_right.tocsr(), g)
    pickle.dump(input_rel.tocsr(), h)
    f.close()
    g.close()
    h.close()

# TODO: configure here
m = open('./data/wn18_leftRel2Right', 'wb')
n = open('./data/wn18_relRight2Left', 'wb')
pickle.dump(leftRel2Right, m)
pickle.dump(relRight2Left, n)
m.close()
n.close()


# obtain a dictionary mapping from entity index to the actual word
# e.g. 14854262 --> "__stool_NN_2"

f = open("./wn18/wordnet-mlj12-definitions.txt", 'r')
data = f.readlines()
f.close()

idx2word = {}
for line in data:
    idx, word, definition = line[:-1].split('\t')
    idx2word[int(idx)] = word

print("Number of words is %s" % len(idx2word))

f = open('./data/entity2word', 'wb')
pickle.dump(idx2word, f)
f.close()


















