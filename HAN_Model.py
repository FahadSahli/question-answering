"""
This code is based on code in:
https://github.com/cairoHy/attention-sum-reader/blob/master/as_reader_tf.py
"""
import random
import sys
import numpy as np
import tensorflow.compat.v1 as tf_compat_v1
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf_compat_v1.disable_eager_execution()

class HAN_Model(object):
    def __init__(self, max_d_length, max_q_length, sess, weight_path, 
                 embedding_matrix, embedding_dim,
                 hidden_size=384, dropout_rate=0.2, two_encoding_layers=False, EPSILON = 10e-8):
        """
        Parameters:
            max_d_length: maximum document length
            max_q_length: maximum query length
            weight_path: path to save and load weights
        """
      
        self.max_d_length = max_d_length
        self.max_q_length = max_q_length
        self.sess = sess
        self.weight_path = weight_path

        self.two_encoding_layers = two_encoding_layers
        self.EPSILON = 10e-8
        self.CAs_len = 10

        # Prepare embeddings
        with tf.device("/gpu:0"):
            self.embedding = tf.Variable(initial_value=embedding_matrix, trainable=False,
                                         name="embedding_matrix_w", dtype="float32")
            
        print("Embedding matrix shape:%d x %d" % (len(embedding_matrix), embedding_dim))

        # Text encoding layer object
        self.text_encoder = self.sequence_encoder(hidden_size=hidden_size, dropout=dropout_rate)

        # Model input and output
        self.q_input = tf_compat_v1.placeholder(dtype=tf.int32, shape=(None, self.max_q_length), name="q_input")
        self.d_input = tf_compat_v1.placeholder(dtype=tf.int32, shape=(None, self.max_d_length), name="d_input")
        self.CAs = tf_compat_v1.placeholder(dtype=tf.int32, shape=(None, self.CAs_len), name="CAs")
        self.y_true = tf_compat_v1.placeholder(shape=(None, self.CAs_len), dtype=tf.float32, name="y_true")

    def build_model(self):

        # Query encoder
        with tf_compat_v1.variable_scope('q_encoder', reuse=tf_compat_v1.AUTO_REUSE):

            # q_embed shape: (None, max_q_length, embedding_dim)
            q_embeddings = tf.nn.embedding_lookup(self.embedding, self.q_input)

            # q_encode shape: (None, hidden_size * 2)
            _, q_encoded = self.apply_inputs(self.text_encoder, q_embeddings)

        # Level 1 (L1) document encoder
        with tf_compat_v1.variable_scope('d_encoder_L1', reuse=tf_compat_v1.AUTO_REUSE):

            # output shape: (None, max_d_length, embedding_dim)
            d_embeddings = tf.nn.embedding_lookup(self.embedding, self.d_input)

            # d_encoded shape: (None, max_d_length, hidden_size * 2)
            d_encoded, _ = self.apply_inputs(self.text_encoder, d_embeddings)

        # Level 1 (L1) attention
        with tf_compat_v1.variable_scope('attention_L1', reuse=tf_compat_v1.AUTO_REUSE):

            # attention_L1 shape: (None, max_d_length)
            attention_L1 = self.att_dot([d_encoded, q_encoded])
            attention_softmax_L1 = tf.nn.softmax(logits=attention_L1, name="attention_softmax_L1")

            # Compute attented document
            # attented_doc shape: (None, max_d_length, hidden_size * 2)
            attented_doc = tf.multiply(tf.expand_dims(attention_softmax_L1, -1), d_encoded, name="attented_doc")

        if(self.two_encoding_layers):
            # Level 2 (L2) document encoder
            with tf_compat_v1.variable_scope('d_encoder_L2', reuse=tf_compat_v1.AUTO_REUSE):

                # d_encoded_L2 shape: (None, max_d_length, hidden_size * 2)
                d_encoded_L2, _ = self.apply_inputs(self.text_encoder, attented_doc)

        # Level 2 (L2) attention
        with tf_compat_v1.variable_scope('attention_L2', reuse=tf_compat_v1.AUTO_REUSE):
          
            # attention_L2 shape = (None, max_d_length)
            attention_L2 = self.att_dot([d_encoded_L2, q_encoded]) if self.two_encoding_layers else self.att_dot([attented_doc, q_encoded])            
            attention_softmax_L2 = tf.nn.softmax(logits=attention_L2, name="softmax_attention_L2")

            # last_prob shape = (None, max_d_length)
            last_prob = tf.multiply(attention_softmax_L2, attention_softmax_L1, name="last_prob")
        

        # Attention Sum
        # y_hat shape = (None, 10)
        self.y_hat = self.sum_probs_batch(self.CAs, self.d_input, last_prob)
        
        # Cross entropy loss function
        output = self.y_hat / tf.reduce_sum(self.y_hat,
                                            axis=len(self.y_hat.get_shape()) - 1, keepdims=True)
        # Compute crossentropy
        epsilon = tf.convert_to_tensor(self.EPSILON, output.dtype.base_dtype, name="epsilon")
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        self.loss = tf.reduce_mean(- tf.reduce_sum(self.y_true * tf.math.log(output),
                                                   axis=len(output.get_shape()) - 1))
        
        self.loss = tf.where(tf.math.is_nan(self.loss), self.EPSILON, self.loss)
        
        # Calculate accuracy
        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(self.y_hat, 1),
                                                                         tf.argmax(self.y_true, 1)), "float")))
        # Model serialization tool
        self.saver = tf_compat_v1.train.Saver()


    def sequence_encoder(self, hidden_size=384, dropout=0.2, merge_mode='concat',
                     recurrent_initializer=tf.initializers.GlorotUniform(), kernel_initializer=tf.initializers.GlorotUniform()):

        """
        forward_layer network processes sequences (words) from 0 to n-1, where n is the number
            of sequences.
            
        backward_layer network processes sequences from n-1 to 0. Output should be reversed 
            to be consistant with 'forward_layer'.
        """
        forward_layer = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, 
                                            go_backwards=False, dropout=dropout,
                                            recurrent_initializer=recurrent_initializer, kernel_initializer=kernel_initializer)
        backward_layer = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, 
                                            go_backwards=True, dropout=dropout,
                                            recurrent_initializer=recurrent_initializer, kernel_initializer=kernel_initializer)
        
        """
        outputf_fw is all hidden states of fw_layer, and its shape is (batch, n, hidden_size).
        state_fw is last state of fw_layer, and its shape is (batch, hidden_size).
        
        outputb_bw is all hidden states of bw_layer, and its shape is (batch, n, hidden_size). outputb_bw[i, j, :]
            corresponds the the jth word of the ith batch. Range of j is from 0 to n-1 where j=0 corresponds 
            to word n-1 of input sequence (e.g., assuming words are indexed from 0 to n-1). 
            
        state_bw is last state of bw_layer, and its shape is (batch, hidden_size). state_bw[i, :] corresponds
            to word 0 of the ith batch because this word is the last to be processed.
        """
        return [forward_layer, backward_layer]

    def apply_inputs(self, layers, inputs):
        """
        layers: a list of two GRUs which are forward_layer and backward_layer
        inputs: either documents or queries
        """

        outputf_fw, state_fw = layers[0](inputs) 
        outputb_bw, state_bw = layers[1](inputs)

        outputb_bw_consistent = tf.reverse(outputb_bw, [1])
        hidden_states = tf.concat([outputf_fw, outputb_bw_consistent], -1)
        
        """
        Return hidden_states and last state. hidden_states[:, -1, :] corresponds to encoding of last word 
            from each batch, and its shape is (batch, 2*hidden_size)
        """
        return hidden_states, hidden_states[:, -1, :]


    def att_dot(self, x):

        documents, queries = x
        
        # res shape = (None, 1, max_d_length)
        res = K.batch_dot(tf.expand_dims(queries, -1), documents, (1, 2))
        return tf.reshape(res, [-1, self.max_d_length])

    # Attention-sum process
    def sum_prob_of_word(self, word_ix, sentence_ixs, sentence_attention_probs):
        word_ixs_in_sentence = tf.where(tf.equal(sentence_ixs, word_ix))
        return tf.reduce_sum(tf.gather(sentence_attention_probs, word_ixs_in_sentence))

    def sum_probs_single_sentence(self, prev, cur):
        candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
        result = tf.scan(
            fn=lambda previous, x: self.sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
            elems=[candidate_indices_i],
            initializer=tf.constant(0., dtype="float32"))
        return result

    def sum_probs_batch(self, candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt):
        result = tf.scan(
            fn=self.sum_probs_single_sentence,
            elems=[candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt],
            initializer=tf.Variable([0] * self.CAs_len, dtype="float32"))
        return result

    def train(self, train_data, valid_data, batch_size=32, epochs=100, opt_name="ADAM",
              lr=0.001, grad_clip=10, beta1=0.9, beta2=0.999):
        
        # Preprocessing the input
        queries, documents, CAs, y_true = self.preprocess_input_sequences(train_data)
        v_queries, v_documents, v_CAs, v_y_true = self.preprocess_input_sequences(valid_data)

        # Define the optimization method of the model
        if opt_name == "SGD":
            optimizer = tf_compat_v1.train.GradientDescentOptimizer(learning_rate=lr)
        elif opt_name == "ADAM":
            optimizer = tf_compat_v1.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
        else:
            raise NotImplementedError("Other Optimizer Not Implemented...")

        # Gradient cropping
        grad_vars = optimizer.compute_gradients(self.loss)
        grad_vars = [
            (tf.clip_by_norm(grad, grad_clip), var)
            if grad is not None else (grad, var)
            for grad, var in grad_vars]

        train_op = optimizer.apply_gradients(grad_vars)
        self.sess.run(tf_compat_v1.global_variables_initializer())

        # Load a previously trained model
        self.load_weight()

        # Prepare validation set data
        v_data = {self.q_input: v_queries,
                  self.d_input: v_documents,
                  self.CAs: v_CAs,
                  self.y_true: v_y_true}

        # early stopping parameter
        best_val_loss, best_val_acc, patience, lose_times = sys.maxsize, 0.0, 5, 0

        # Start training
        corrects_in_epoch, loss_in_epoch = 0, 0
        batches, v_batches = len(queries) // batch_size, len(v_queries) // batch_size
        batch_idx, v_batch_idx = np.random.permutation(batches), np.arange(v_batches)
        print("Train on {} batches, {} samples per batch.".format(batches, batch_size))
        print("Validate on {} batches, {} samples per batch.".format(v_batches, batch_size))

        for step in range(batches * epochs):
            # End of an Epoch, output log and shuffle
            if step % batches == 0:
                corrects_in_epoch, loss_in_epoch = 0, 0
                print("--------Epoch : {}".format(step // batches + 1))
                np.random.shuffle(batch_idx)

            # Get the data for the next batch
            slices = np.index_exp[
                     batch_idx[step % batches] * batch_size:(batch_idx[step % batches] + 1) * batch_size]
            data = {self.q_input: queries[slices],
                    self.d_input: documents[slices],
                    self.CAs: CAs[slices],
                    self.y_true: y_true[slices]}

            # Train, update parameters, output current accuracy of Epoch
            loss_, _, corrects_in_batch = self.sess.run([self.loss, train_op, self.correct_prediction],
                                                        feed_dict=data)
            corrects_in_epoch += corrects_in_batch
            loss_in_epoch += loss_ * batch_size
            samples_in_epoch = (step % batches + 1) * batch_size
            print("Trained samples in this epoch : {}".format(samples_in_epoch))
            print("Step : {}/{}.\nLoss : {:.4f}.\nAccuracy : {:.4f}".format(step % batches,
                                                                                   batches,
                                                                                   loss_in_epoch / samples_in_epoch,
                                                                                   corrects_in_epoch / samples_in_epoch))

            # Save the model every 200 steps and use the validation set to calculate the accuracy rate and determine whether it is early stop
            if step % 200 == 0 and step != 0:
                # Due to insufficient GPU memory, it is still calculated as batch
                val_samples, val_corrects, v_loss = 0, 0, 0
                for i in range(v_batches):

                    start = v_batch_idx[i % v_batches] * batch_size
                    stop = (v_batch_idx[i % v_batches] + 1) * batch_size
                    _v_slice = np.index_exp[start:stop]
                    v_data = {self.q_input: v_queries[_v_slice],
                              self.d_input: v_documents[_v_slice],
                              self.CAs: v_CAs[_v_slice],
                              self.y_true: v_y_true[_v_slice]}

                    loss_, v_correct = self.sess.run([self.loss, self.correct_prediction], feed_dict=v_data)
                    val_samples = val_samples + batch_size
                    val_corrects = val_corrects + v_correct
                    v_loss = v_loss + loss_ * batch_size

                val_acc = val_corrects / val_samples
                val_loss = v_loss / val_samples
                print("Val acc : {:.4f}".format(val_acc))
                print("Val Loss : {:.4f}".format(val_loss))

                if val_acc > best_val_acc or val_loss < best_val_loss:
                    # Save a better model
                    lose_times = 0
                    best_val_loss, best_val_acc = val_loss, val_acc
                    path = self.saver.save(self.sess,
                                           self.weight_path + \
                                           'machine_reading-val_acc-{:.4f}-val_loss-{:.4f}.model'.format(val_acc, val_loss),
                                           global_step=step)
                    print("Save model to {}.".format(path))

                else:
                    lose_times += 1
                    print("Lose_time/Patience : {}/{} .".format(lose_times, patience))
                    if lose_times >= patience:
                        print("Stop training.".format(lose_times, patience))
                        exit(0)

    def test(self, test_data, batch_size):
        
        queries, documents, CAs, y_true = self.preprocess_input_sequences(test_data)
        print("Test on {} samples, {} per batch.".format(len(queries), batch_size))

        # Load a previously trained model
        self.load_weight()
        
        # Testing
        batches = len(queries) // batch_size
        batch_idx = np.arange(batches)
        correct_num, total_num = 0, 0
        for i in range(batches):
            start = batch_idx[i % batches] * batch_size
            stop = (batch_idx[i % batches] + 1) * batch_size
            slices = np.index_exp[start:stop]
            data = {self.q_input: queries[slices],
                    self.d_input: documents[slices],
                    self.CAs: CAs[slices],
                    self.y_true: y_true[slices]}
                     
            correct, = self.sess.run([self.correct_prediction], feed_dict=data)
            
            correct_num, total_num = correct_num + correct, total_num + batch_size
            
        test_acc = correct_num / total_num
        print("Test accuracy is : {:.5f}".format(test_acc))

    def load_weight(self):

        ckpt = tf.train.get_checkpoint_state(self.weight_path)
        if ckpt is not None:
            print("Load model from {}.".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No previous models.")

    @staticmethod
    def union_shuffle(data):
        q, d, CAs, a = data
        c = list(zip(q, d, CAs, a))
        random.shuffle(c)
        return zip(*c)

    def preprocess_input_sequences(self, data, shuffle=True):
        """
        Preprocessing input asï¼š
          shuffle
          PAD To a fixed-length sequence
          y_true is a vector of length self.CAs_len, index = 0 is the correct answer, and one-hot encoding
        """
        queries, documents, CAs, _ = self.union_shuffle(data) if shuffle else data
        d_lens = [len(i) for i in documents]

        queries = pad_sequences(queries, maxlen=self.max_q_length, dtype="int32", padding="post", truncating="post")
        documents = pad_sequences(documents, maxlen=self.max_d_length, dtype="int32", padding="post", truncating="post")
        CAs = pad_sequences(CAs, maxlen=self.CAs_len, dtype="int32", padding="post", truncating="post")
        y_true = np.zeros_like(CAs)
        y_true[:, 0] = 1
        return queries, documents, CAs, y_true