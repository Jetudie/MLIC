import numpy as np
import json
import time
import math

class naive_bayes:
    def __init__(self, file_allwords, file_comment, file_rating):
    # Load files and top_k words
        self.comment = np.load(file_comment)
        self.rating = np.load(file_rating)
        with open(file_allwords,"r") as read_file:
            self.words_all = json.load(read_file)

    def set_top_k(self, top_k): 
    # Calculate the top k words' frequency
        self.top_k = top_k
        self.words_top_k = [None]*self.top_k
        self.freq = [ [ 0 for i in range(self.top_k) ] for j in range(2) ]
        self.prior_positive = 0
        self.prior_negative = 0
        self.likelihood_positive = [None]*self.top_k
        self.likelihood_negative = [None]*self.top_k
        for key, val in self.words_all.items():
            if val <= self.top_k:
                self.words_top_k[val-1] = key
        print("Top", self.top_k, "words used")
        n_comment = len(self.comment)
        print("Training with", n_comment, "comments")
        start_time = time.time()
        # search for words in comment
        # store counts in self.freq["rating"]["words"]
        for c in range(n_comment):
            for n in range(self.top_k):
                if n+1 in self.comment[c]:
                    self.freq[self.rating[c]][n] += 1
        stop_time = time.time()
        print(self.freq)
        print("Time:", stop_time-start_time, "sec")
        print("Calculating prior, likelyhood")
        pos_sum = sum(self.freq[1])
        neg_sum = sum(self.freq[0])
        self.prior_positive = math.log(pos_sum) - math.log(pos_sum+neg_sum)
        self.prior_negative = math.log(neg_sum) - math.log(pos_sum+neg_sum)
        pos_sum_log = math.log(pos_sum)
        neg_sum_log = math.log(neg_sum)
        for n in range(len(self.freq[1])):
            freq = self.freq[1][n] + (self.freq[1][n] == 0)*0.000001
            self.likelihood_positive[n] = math.log(freq) - pos_sum_log
        for n in range(len(self.freq[0])):
            freq = self.freq[0][n] + (self.freq[0][n] == 0)*0.000001
            self.likelihood_negative[n] = math.log(freq) - neg_sum_log

    def test(self, file_test_comment, file_test_rating): # test the comment file
        comment_test = np.load(file_test_comment)
        rating_test = np.load(file_test_rating)
        n_comment = len(comment_test)
        self.result = [None]*n_comment
        print("Using top", self.top_k, "words.")
        print("Testing", n_comment, "comment.")
        # Compare the possibilities and store the result for each comment
        start_time = time.time()
        for c in range(n_comment):
            self.result[c] = 1 if self.test_comment(comment_test[c]) else 0
        self.performance(rating_test)
        stop_time = time.time()
        print("Time:", stop_time-start_time, "sec\n")

    def test_comment(self, test_comment): 
        # test a comment by calculating positive/negative probabilities and compare them
        words = []
        posterior_negative = 0
        posterior_positive = 0
        likelihood_neg = 0
        likelihood_pos = 0
        # Search exist words in the comment
        for n in range(self.top_k):
            if n+1 in test_comment:
                words.append(n)
        for w in words:
            likelihood_neg += self.likelihood_negative[w]
            likelihood_pos += self.likelihood_positive[w]
        posterior_negative = likelihood_neg + self.prior_negative
        posterior_positive = likelihood_pos + self.prior_positive
        return (posterior_positive > posterior_negative)

    def performance(self, test_rating):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for n in range(len(test_rating)):
            if self.result[n] == test_rating[n]:
                if self.result[n] == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if self.result[n] == 1:
                    false_positive += 1
                else:
                    false_negative += 1

        accuracy = (true_positive + true_negative) / (true_positive + false_negative)
        accuracy1 = (true_positive + true_negative) / len(self.result)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        print("Accuracy=", accuracy)
        print("Accuracy1=", accuracy1)
        print("Precision=", precision)
        print("Recall=", recall)

if __name__ == "__main__":
    FILE_WORDS = "./imdb_word_index.json"
    FILE_COMMENT = "./imdb/x_train.npy"
    FILE_RATING = "./imdb/y_train.npy"

    FILE_TEST_COMMENT = "./imdb/x_test.npy"
    FILE_TEST_RATING = "./imdb/y_test.npy"
    test = naive_bayes(FILE_WORDS,FILE_COMMENT,FILE_RATING)

    test.set_top_k(100)
    test.test(FILE_TEST_COMMENT, FILE_TEST_RATING)
    print("\n")
    '''
    test.set_top_k(1000)
    test.test(FILE_TEST_COMMENT, FILE_TEST_RATING)
    print("\n")

    test.set_top_k(10000)
    test.test(FILE_TEST_COMMENT, FILE_TEST_RATING)
    '''