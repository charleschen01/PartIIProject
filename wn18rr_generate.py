
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


training_set = read_triplets("./wordnet/wordnet-mlj12-train.txt")
test_set = read_triplets("./wordnet/wordnet-mlj12-test.txt")
validation_set = read_triplets("./wordnet/wordnet-mlj12-valid.txt")


# remove the items that contain the above relations


filtered_training_set = [triplet for triplet in training_set if triplet[1] not in relations_to_remove]
filtered_test_set = [triplet for triplet in test_set if triplet[1] not in relations_to_remove]
filtered_validation_set = [triplet for triplet in validation_set if triplet[1] not in relations_to_remove]

print("Training set size: {} -> {}".format(len(training_set), len(filtered_training_set)))
print("Test set size: {} -> {}".format(len(test_set), len(filtered_test_set)))
print("Validation set size: {} -> {}".format(len(validation_set), len(filtered_validation_set)))


# outputs the three sets into text file


def write_triplets(path, dataset):
    with open(path, 'w') as f:
        for lhs, rel, rhs in dataset:
            f.write("{}\t{}\t{}\n".format(lhs, rel, rhs))


write_triplets("./wn18rr/wn18rr_train.txt", filtered_training_set)
write_triplets("./wn18rr/wn18rr_test.txt", filtered_test_set)
write_triplets("./wn18rr/wn18rr_valid.txt", filtered_validation_set)




