from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
import numpy as np
from Classifier import Classifier

def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text


if __name__ == '__main__':

    stemmer = EnglishStemmer()

    knn_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__n_neighbors': [3, 5],
    }

    svm_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__C': np.array([1,10,100,1000]),
        'classifier__gamma': np.array([0.1, 0.25]),

    }

    classifier = Classifier(knn_parameters, classifier_flag = "knn")
    classifier.create_Training_Test_data("./Ham_Spam_comments/Training", "./Ham_Spam_comments/Test")
    classifier.fit_grid_search()
    classifier.predict_grid_search()