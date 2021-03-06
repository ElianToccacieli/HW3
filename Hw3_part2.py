from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
import numpy as np
from Classifier import Classifier
np.seterr(divide='ignore', invalid='ignore')

stemmer = EnglishStemmer()


def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text


if __name__ == '__main__':


    knn_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__n_neighbors': [3, 5,7,9],
    }


    C = np.array([10**i for i in range(-2,3)])

    svm_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__C': C,
        'classifier__kernel': ('rbf','linear','poly'),
        'classifier__gamma': np.array([ 0.1, 0.25, 1]),

    }

    nb_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
    }


    print("Starting K-nearest neighbors")

    knn_classifier = Classifier(knn_parameters, classifier_flag = "knn")
    knn_classifier.create_Training_Test_data("./Positve_negative_sentences/Training", "./Positve_negative_sentences/Test")
    knn_classifier.fit_grid_search()
    knn_classifier.predict_grid_search()

    print()
    print("----------------------")
    print("Starting Support Vector Machine")
    print("----------------------")
    print()

    svm_classifier = Classifier(svm_parameters, classifier_flag="svm")
    svm_classifier.create_Training_Test_data("./Positve_negative_sentences/Training","./Positve_negative_sentences/Test")
    svm_classifier.fit_grid_search()
    svm_classifier.predict_grid_search()

    print()
    print("----------------------")
    print("Starting Naive Bayes classifier")
    print("----------------------")
    print()

    nb_classifier = Classifier(nb_parameters, classifier_flag="nb")
    nb_classifier.create_Training_Test_data("./Positve_negative_sentences/Training",
                                             "./Positve_negative_sentences/Test")
    nb_classifier.fit_grid_search()
    nb_classifier.predict_grid_search()

