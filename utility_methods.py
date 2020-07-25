import numpy as np
import os
from tensorflow.python.platform import gfile
from functools import reduce

# end sentence tag
EoS_ID = 2

def paths_to_processed_data(data_dir, train_file, valid_file, test_file, vocab_size, output_dir):
    """
    This method takes paths to all data files and generates paths to corresponding ID files
    """

    idx_train_file = os.path.join(data_dir, output_dir, train_file + ".%d.id.txt" % vocab_size)
    idx_valid_file = os.path.join(data_dir, output_dir, valid_file + ".%d.id.txt" % vocab_size)
    idx_test_file = os.path.join(data_dir, output_dir, test_file + ".%d.id.txt" % vocab_size)
    vocab_file = os.path.join(data_dir, output_dir, "vocab.%d.txt" % vocab_size)

    return vocab_file, idx_train_file, idx_valid_file, idx_test_file

def read_processed_data(file):
    """
    This method Reads data file in ID format
    """

    documents, questions, answers, candidates = [], [], [], []

    with gfile.GFile(file, mode="r") as f:
        counter = 0
        d, q, a, A = [], [], [], []
        for line in f:
            counter += 1

            if counter % 100000 == 0:
                print("Reading line %d in %s" % (counter, file))

            if counter % 22 == 21:

                tmp = line.strip().split("\t")
                q = tmp[0].split(" ") + [EoS_ID]
                a = [1 if tmp[1] == i else 0 for i in d]

                A = [a for a in tmp[2].split("|")]

                # Put the correct answer first
                A.remove(tmp[1])
                A.insert(0, tmp[1])

            elif counter % 22 == 0:
                documents.append(d)
                questions.append(q)
                answers.append(a)
                candidates.append(A)

                d, q, a, A = [], [], [], []

            else:
                # Add EoS ID at the end of each sentence
                d_tem = [i for i in line.strip().split(" ") if i != '']
                d.extend(d_tem + [EoS_ID])

    d_lens = [len(i) for i in documents]
    q_lens = [len(i) for i in questions]

    avg_d_len = reduce(lambda x, y: x + y, d_lens) / len(documents)
    print("Document average length: %d." % avg_d_len)
    print("Document midden length: %d." % len(sorted(documents, key=len)[len(documents) // 2]))

    avg_q_len = reduce(lambda x, y: x + y, q_lens) / len(questions)
    print("Question average length: %d." % avg_q_len)
    print("Question midden length: %d." % len(sorted(questions, key=len)[len(questions) // 2]))
    
    return questions, documents, candidates, answers

def gen_embeddings(word_dict, embed_dim, in_file=None, init=np.zeros):
    """
    Tjis method creates an initialized word vector matrix for the vocabulary.
        If a word is not in the word vector file, a vector will be initialized randomly.
    
    word_dict: word to id mapping
    embed_dim: the dimensions of the word vector.
    in_file: pre-trained word vector file. 
    init: how to initialize the words not found in the pre-training file
    """
    num_words = max(word_dict.values()) + 1
    embedding_matrix = init(-0.1, 0.1, (num_words, embed_dim))
    print('Embeddings: %d x %d' % (num_words, embed_dim))

    if not in_file:
        return embedding_matrix

    assert get_dim(in_file) == embed_dim
    print('Loading embedding file: %s' % in_file)

    pre_trained = 0
    for line in open(in_file):
        sp = line.split()
        if sp[0] in word_dict:
            pre_trained += 1
            embedding_matrix[word_dict[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype=np.float32)

    print('Pre-trained: %d (%.2f%%)' %
                 (pre_trained, pre_trained * 100.0 / num_words))
    return embedding_matrix


def get_dim(in_file):

    # get length of first vector
    line = gfile.GFile(in_file, mode='r').readline()
    return len(line.split()) - 1

def get_max_length(lines):
    lens = [len(line) for line in lines]
    return max(lens)

def load_vocab(vocab_file):
    """
    This method loads vocab_file. It returns a word_dict with entries as {word : its ID}

    vocab_file: path to vocab file
    """

    if not gfile.Exists(vocab_file):
        raise ValueError("Vocabulary file %s not found.", vocab_file)

    word_dict = {}
    word_id = 0

    with gfile.GFile(vocab_file, "r") as f:
        for line in f:

            # Line has a single word with trailing new line char which needs to be removed
            word_dict.update({line.strip(): word_id})
            word_id += 1

    return word_dict