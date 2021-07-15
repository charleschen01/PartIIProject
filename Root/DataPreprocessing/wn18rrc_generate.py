
# we want to remove the following relations

relations_to_remove = [
    '_member_of_domain_topic',
    '_synset_domain_usage_of',
    '_instance_hyponym',
    '_hyponym',
    '_member_holonym',
    '_synset_domain_region_of',
    '_part_of'
]


def read_triplets(path):
    triplet_set = []
    with open(path, 'r') as f:
        for triplet in f.readlines():
            lhs, rel, rhs = triplet[:-1].split('\t')
            triplet_set.append((lhs, rel, rhs))
    return triplet_set


# load datasets from the original txt files
training_set = read_triplets("./wn18/wordnet-mlj12-train.txt")
test_set = read_triplets("./wn18/wordnet-mlj12-test.txt")
validation_set = read_triplets("./wn18/wordnet-mlj12-valid.txt")


# remove the items that contain the above relations


filtered_training_set = [triplet for triplet in training_set if triplet[1] not in relations_to_remove]
filtered_test_set = [triplet for triplet in test_set if triplet[1] not in relations_to_remove]
filtered_validation_set = [triplet for triplet in validation_set if triplet[1] not in relations_to_remove]

print("Training set size: {} -> {}".format(len(training_set), len(filtered_training_set)))
print("Test set size: {} -> {}".format(len(test_set), len(filtered_test_set)))
print("Validation set size: {} -> {}".format(len(validation_set), len(filtered_validation_set)))

# do a second filtering on validation set and test set
train_synsets = set()
for triplet in filtered_training_set:
    train_synsets.add(triplet[0])
    train_synsets.add(triplet[2])
filtered_test_set = [triplet for triplet in filtered_test_set if triplet[0] in train_synsets and
                                                                 triplet[2] in train_synsets]
filtered_validation_set = [triplet for triplet in filtered_validation_set if triplet[0] in train_synsets and
                                                                             triplet[2] in train_synsets]

print("After second round of filtering")
print("Size of synsets: {}".format(len(train_synsets)))
print("Total size: {}".format(len(filtered_training_set)+len(filtered_validation_set)+len(filtered_test_set)))
print("Training set size: {}".format(len(filtered_training_set)))
print("Test set size: {}".format(len(filtered_test_set)))
print("Validation set size: {}".format(len(filtered_validation_set)))

# outputs the three sets into text file


def write_triplets(path, dataset):
    with open(path, 'w') as f:
        for lhs, rel, rhs in dataset:
            f.write("{}\t{}\t{}\n".format(lhs, rel, rhs))


write_triplets("./wn18rr/wn18rr_train.txt", filtered_training_set)
write_triplets("./wn18rr/wn18rr_test.txt", filtered_test_set)
write_triplets("./wn18rr/wn18rr_valid.txt", filtered_validation_set)