# create dataset files for wn18_subset and others
import numpy as np
import pickle
import scipy.sparse as sp

entitylist = []
relationlist = []

for datatype in ['train', 'valid', 'test']:
    f = open("./wn18_subset/wn18_%s_subset.txt" % datatype, 'r')
    data = f.readlines()
    f.close()

    # obtain a set of entities and relations
    for triplet in data:
        lhs, rel, rhs = triplet[:-1].split('\t')
        entitylist += [lhs, rhs]
        relationlist += [rel]

entityset = np.sort(list(set(entitylist)))
relset = np.sort(list(set(relationlist)))

print("{} entities".format(len(entityset)))
print("{} relations".format(len(relset)))


# we now fill in the dictionaries

entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}

entity_id = 0
for i in entityset:
    entity2id[i] = entity_id
    id2entity[id] = i
    entity_id += 1


relation_id = 0
for i in relset:
    relation2id[i] = relation_id
    id2relation[relation_id] = i
    relation_id += 1

f1 = open('./data/wn18_subset_entity2id', 'wb')
f2 = open('./data/wn18_subset_id2entity', 'wb')
f3 = open('./data/wn18_subset_relation2id', 'wb')
f4 = open('./data/wn18_subset_id2relation', 'wb')
pickle.dump(entity2id, f1)
pickle.dump(id2entity, f2)
pickle.dump(relation2id, f3)
pickle.dump(id2relation, f4)
f1.close()
f2.close()
f3.close()
f4.close()

# create dataset files

for datatype in ['train', 'valid', 'test']:
    f = open("./wn18_subset/wn18_%s_subset.txt" % datatype, 'r')
    data = f.readlines()
    f.close()

    input_left = sp.lil_matrix((entity_id, len(data)), dtype='float32')
    input_right = sp.lil_matrix((entity_id, len(data)), dtype='float32')
    input_rel = sp.lil_matrix((relation_id, len(data)), dtype='float32')

    # fill the sparse matrices
    j = 0
    for triplet in data:
        lhs, rel, rhs = triplet[:-1].split('\t')
        input_left[entity2id[lhs], j] = 1
        input_right[entity2id[rhs], j] = 1
        input_rel[relation2id[rel], j] = 1
        j += 1

    f = open('./data/wn18_subset_%s_lhs' % datatype, 'wb')
    g = open('./data/wn18_subset_%s_rhs' % datatype, 'wb')
    h = open('./data/wn18_subset_%s_rel' % datatype, 'wb')
    pickle.dump(input_left.tocsr(), f)
    pickle.dump(input_right.tocsr(), g)
    pickle.dump(input_rel.tocsr(), h)
    f.close()
    g.close()
    h.close()


















