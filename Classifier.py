import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pprint as pp

class Classifier(object):
    def __init__(self,parameters, classifier_flag = "svm"):
        self.parameters = parameters
        self.inizialize_vectorizer()
        self.inizialize_classifier(classifier_flag)
        self.inizialize_pipeline()

    def inizialize_vectorizer(self):
        self.vectorizer = TfidfVectorizer(strip_accents=None,
                                     preprocessor=None,
                                     )

    def inizialize_classifier(self, flag):
        if flag == "knn":
            self.classifier = KNeighborsClassifier()
        elif flag == "svm":
            self.classifier = svm.SVC()

        elif flag == "nb":
            self.classifier = MultinomialNB()

    def inizialize_pipeline(self):
        self.pipeline = Pipeline([
            ('vect', self.vectorizer),
            ('classifier', self.classifier),
        ])

    def create_Training_Test_data(self, data_folder_training_set, data_folder_test_set):

        training_dataset = load_files(data_folder_training_set)
        test_dataset = load_files(data_folder_test_set)
        print()
        print("----------------------")
        print(training_dataset.target_names)
        print("----------------------")
        print()
        # Load Training-Set
        self.X_train, X_test_DUMMY_to_ignore, self.Y_train, Y_test_DUMMY_to_ignore = train_test_split(training_dataset.data,
                                                                                            training_dataset.target,
                                                                                            test_size=0.0)
        self.target_names = training_dataset.target_names

        # Load Test-Set
        X_train_DUMMY_to_ignore, self.X_test, Y_train_DUMMY_to_ignore, self.Y_test = train_test_split(test_dataset.data,
                                                                                            test_dataset.target,
                                                                                            train_size=0.0)

        target_names = training_dataset.target_names
        print()
        print("----------------------")
        print("Creating Training Set and Test Set")
        print()
        print("Training Set Size")
        print(self.Y_train.shape)
        print()
        print("Test Set Size")
        print(self.Y_test.shape)
        print()
        print("Classes:")
        print(target_names)
        print("----------------------")


    def fit_grid_search(self):
        self.grid_search = GridSearchCV(self.pipeline,
                                   self.parameters,
                                   scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                                   cv=10,
                                   n_jobs=4,
                                   verbose=0)

        ## Start an exhaustive search to find the best combination of parameters considering the select scoring-function,
        ## on the entire original TRAINING-Set
        print()
        self.grid_search.fit(self.X_train, self.Y_train)
        print()
        ## Print results for each combination of parameters.
        number_of_candidates = len(self.grid_search.cv_results_['params'])
        print("Results:")
        for i in range(number_of_candidates):
            print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
                  (self.grid_search.cv_results_['params'][i],
                   self.grid_search.cv_results_['mean_test_score'][i],
                   self.grid_search.cv_results_['std_test_score'][i]))

        print()
        print("Best Estimator:")
        pp.pprint(self.grid_search.best_estimator_)
        print()
        print("Best Parameters:")
        pp.pprint(self.grid_search.best_params_)
        print()
        print("Used Scorer Function:")
        pp.pprint(self.grid_search.scorer_)
        print()
        print("Number of Folds:")
        pp.pprint(self.grid_search.n_splits_)
        print()

    def predict_grid_search(self):
        ## Let's train the classifier that achieved the best performance,
        ## according to the selected scoring-function.
        Y_predicted = self.grid_search.predict(self.X_test)
        ## Evaluate the performance of the classifier on the original Test-Set
        output_classification_report = metrics.classification_report(
            self.Y_test,
            Y_predicted,
            target_names=self.target_names)
        print()
        print()
        "----------------------------------------------------"
        print(output_classification_report)
        print()
        "----------------------------------------------------"
        print()

        ## Compute the confusion matrix
        confusion_matrix = metrics.confusion_matrix(self.Y_test, Y_predicted)
        print()
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)
        print()
        # accuracy
        acc = metrics.accuracy_score(self.Y_test, Y_predicted)
        # compute the matthew coefficent
        matthew_coeff = metrics.matthews_corrcoef(self.Y_test, Y_predicted)
        print("accuracy_score: ", acc)
        print("matthew coefficient: ", matthew_coeff)




