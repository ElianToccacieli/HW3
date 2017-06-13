from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
from Classifier import Classifier

stemmer = EnglishStemmer()

def stemming_tokenizer(text):
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text

if __name__ == '__main__':


    knn_parameters = {
        'vect__tokenizer': [None, stemming_tokenizer],
        'vect__stop_words': [None, 'english'],
        'vect__ngram_range': [(1, 1), (1, 2), ],
        'classifier__n_neighbors': [3, 5],
    }


    classifier = Classifier(knn_parameters, classifier_flag = "knn")
    classifier.create_Training_Test_data("./Ham_Spam_comments/Training", "./Ham_Spam_comments/Test")
    classifier.fit_grid_search()
    classifier.predict_grid_search()