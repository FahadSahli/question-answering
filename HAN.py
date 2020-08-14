from Config import Config
from HAN_Model import HAN_Model
import utility_methods
import tensorflow.compat.v1 as tf_compat_v1
import numpy as np

def prepare_data():
    # Directories to data
    data_dir = config.data_dir + config.data_type + "_data/"
    train_file = config.data_type + "_train"
    valid_file = config.data_type + "_valid_2000ex"
    test_file = config.data_type + "_test_2500ex"

    # Directory to store processed data
    output_dir = "processedData"

    vocab_size = 100000
    
    vocab_file, idx_train_file, idx_valid_file, idx_test_file = utility_methods.paths_to_processed_data(
        data_dir, train_file, valid_file, test_file, vocab_size, output_dir)

    train_data = utility_methods.read_processed_data(idx_train_file)
    val_data = utility_methods.read_processed_data(idx_valid_file)
    test_data = utility_methods.read_processed_data(idx_test_file)
    
    max_d_length = utility_methods.get_max_length(train_data[1])
    max_q_length = utility_methods.get_max_length(train_data[0])

    return vocab_file, train_data, val_data, test_data, max_d_length, max_q_length

if __name__ == '__main__':
    config = Config()

    vocab_file, train_data, val_data, test_data, max_d_length, max_q_length = prepare_data()

    word_dict = utility_methods.load_vocab(vocab_file)
    embedding_matrix = utility_methods.gen_embeddings(word_dict, config.embedding_dim, config.embedding_file, init=np.random.uniform)

    sess = tf_compat_v1.Session()
    model = HAN_Model(max_d_length, max_q_length, sess, config.weight_path,
                        embedding_matrix, config.embedding_dim,
                        hidden_size=config.hidden_size, dropout_rate=config.dropout_rate, 
                        two_encoding_layers=config.two_encoding_layers)

    model.build_model()

    if config.training:
        model.train(train_data=train_data,
                    valid_data=val_data,
                    batch_size=config.batch_size,
                    epochs=config.num_epoches,
                    opt_name=config.optimizer,
                    lr=config.learning_rate,
                    grad_clip=config.grad_clipping,
                    beta1=config.beta1)
    if config.testing:
        model.test(test_data, config.batch_size)