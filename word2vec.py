from collections import OrderedDict
from nltk.tokenize import sent_tokenize
import numpy as np


class Word2VecDeepNeuralNet:
    def __init__(self, n_features, learning_rate, window_size, file_path, epochs=100):
        self.file_path = file_path
        self.build_vocab()
        self.W1 = np.random.randn(len(self.vocab), n_features)
        self.W2 = np.random.randn(n_features, len(self.vocab))
        self.alpha = learning_rate
        self.window_size = window_size
        self.epochs = 100
        # self.text_list = text_list

    def forward_prop(self, input_word):
        self.W1_linear_out = np.dot(self.W1.T, input_word)
        self.W2_linear_out = np.dot(self.W2.T, self.W1_linear_out)
        softmax_output = np.exp(self.W2_linear_out)/np.sum(np.exp(self.W2_linear_out))
        return softmax_output

    def process(self):
        for _ in range(self.epochs):
            for sentence in self.text_list:
                word_list = sentence.split()
                for i in range(len(word_list)):
                    # one_hot_encoded_center_word = get_one_hot_encoded_word(vocab_len, word_list[i])
                    for j in range(-len(self.window_size)/2, len(self.window_size)/2+1):
                        if j == 0:
                            continue
                        index = i + j
                        if index < 0:
                            continue
                        elif index >= len(word_list):
                            continue
                        else:
                            one_hot_encoded_center_word = get_one_hot_encoded_word(len(self.vocab), word_list[i])
                            softmax_output = self.forward_prop(one_hot_encoded_center_word)
                             # pass center and context words
                            self.back_prop_grad_update(softmax_output, word_list[i], word_list[j])

    def back_prop_grad_update(self, softmax_output, center_word, context_word):
        y_truth = self.get_one_hot_encoded_word(context_word)
        index_context = list(self.vocab.keys()).index(context_word)
        index_center = list(self.vocab.keys()).index(center_word)
        self.W1[index_center, :] -= self.alpha * (self.W2[:, index_context] - np.sum(self.W2 * softmax_output.T, axis=1))
        self.W2[:, index_center] -= self.alpha * (self.W1[index_center, :] - np.sum(self.W1.T * softmax_output.T, axis=1))

    def get_one_hot_encoded_word(self, word):
        one_hot_encoded_word = np.zeros((len(self.vocab), 1))
        index_of_word = list(self.vocab.keys()).index('word')
        one_hot_encoded_word[index_of_word] = 1
        return one_hot_encoded_word

    def build_vocab(self):
        file = open(self.file_path, 'r')
        vocab_unordered = {}
        self.vocab = OrderedDict()
        sentences = sent_tokenize(' '.join(file.readlines()))
        self.text_list = []
        for sent in sentences:
            self.text_list.append(sent[:-1])

        for sentence in self.text_list:
            for word in sentence.split():
                if(word not in vocab_unordered.keys()):
                    vocab_unordered[word] = 1
                else:
                    vocab_unordered[word] += 1

        vocab_list = sorted(vocab_unordered.items(), key=lambda x: x[1], reverse=True)

        for tup in vocab_list:
            self.vocab[tup[0]] = tup[1]


if __name__ == "__main__":
    path = 'C:/GitProjects/Word2Vec Python Implementation/input.txt'
    my_nlp_nn = Word2VecDeepNeuralNet(50, 0.001, 4, path)
    print(my_nlp_nn.W1)
    print(my_nlp_nn.W2)
