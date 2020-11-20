def generate_subset(train_path, test_path, valid_path, train_subset_path, test_subset_path, valid_subset_path):
    with open(train_path, 'r') as f:
        train_data = f.readlines()

    # use sets to record down the entities and relations
    entity_set = set()
    relation_set = set()

    # initialise subsets
    train_subset = train_data[0:10000]
    test_subset = []
    valid_subset = []

    # record down entities and relations
    for triplet in train_subset:
        lhs, rel, rhs = triplet[:-1].split('\t')
        entity_set.add(lhs)
        entity_set.add(rhs)
        relation_set.add(rel)

    # create training and validation subsets
    # all entities/relations appearing in the valid/test subsets should occur in train subset
    for path in [test_path, valid_path]:
        with open(path, 'r') as f:
            data = f.readlines()
        for triplet in data:
            lhs, rel, rhs = triplet[:-1].split('\t')
            if lhs in entity_set and rhs in entity_set and rel in relation_set:
                if path == test_path:
                    test_subset.append(triplet)
                else:
                    valid_subset.append(triplet)
    print("Test subset contains %d items" % len(test_subset))
    print("Valid subset contains %d items" % len(valid_subset))

    # output subsets as text file
    with open(train_subset_path, 'w') as f:
        for line in train_subset:
            f.write(line)
    with open(test_subset_path, 'w') as f:
        for line in test_subset:
            f.write(line)
    with open(valid_subset_path, 'w') as f:
        for line in valid_subset:
            f.write(line)


print("Generate subsets for WN18: ")
generate_subset("./wn18/wordnet-mlj12-train.txt",
                "./wn18/wordnet-mlj12-test.txt",
                "./wn18/wordnet-mlj12-valid.txt",
                "./wn18_subset/wn18_train_subset.txt",
                "./wn18_subset/wn18_test_subset.txt",
                "./wn18_subset/wn18_valid_subset.txt")

print("Generate subsets for WN18RR: ")
generate_subset("./wn18rr/wn18rr_train.txt",
                "./wn18rr/wn18rr_test.txt",
                "./wn18rr/wn18rr_valid.txt",
                "./wn18rr_subset/wn18rr_train_subset.txt",
                "./wn18rr_subset/wn18rr_test_subset.txt",
                "./wn18rr_subset/wn18rr_valid_subset.txt")






