import random
from warnings import filterwarnings

import jieba
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def load_raw_dataset(path):
    # dataset = []
    texts = []
    labels = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            label, text = line.strip("\n").split("\t")
            # dataset.append([text, label])
            texts.append(text)
            labels.append(label)

    # return dataset
    return (texts, labels)


def load_stopwords(path):
    stopwords = []
    with open(path, "r", encoding="utf8") as f:
        for word in f:
            stopwords.append(word.strip("\n"))

    return stopwords


def load_label_dict(path):
    """
    Returns:
        (label2ix, ix2label)
    """

    label2ix = {}
    ix2label = []
    with open(path, "r", encoding="utf8") as f:
        for label in f:
            label = label.strip("\n")
            label2ix[label] = len(label2ix)
            ix2label.append(label)

    return (label2ix, ix2label)


if __name__ == "__main__":
    filterwarnings("ignore")
    random.seed(1)
    np.random.seed(1)

    print("Load stopwords...")
    stopwords = load_stopwords("./data/stopword.txt")

    print("Load label dict...")
    label2ix, ix2label = load_label_dict("./data/label_dict.txt")

    print("Load train data...")
    train_dataset = load_raw_dataset("./data/news/cnews.train.txt")
    random.shuffle(train_dataset)

    print("Load test data...")
    test_dataset = load_raw_dataset("./data/news/cnews.test.txt")

    train_texts, train_labels = map(list, zip(*train_dataset))
    test_texts, test_labels = map(list, zip(*test_dataset))

    train_labels = np.array([label2ix[i] for i in train_labels])
    test_labels = np.array([label2ix[i] for i in test_labels])

    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, tokenizer=jieba.cut, decode_error="ignore")

    print("Transform train vectors...")
    train_vectors = tfidf_vectorizer.fit_transform(train_texts)

    print("Transform test vectors...")
    test_vectors = tfidf_vectorizer.transform(test_texts)

    print("Save train vectors...")
    sparse.save_npz("./data/train_vectors.npz", train_vectors)

    print("Save train labels...")
    np.save("./data/train_labels.npy", train_labels)

    print("Save test vectors...")
    sparse.save_npz("./data/test_vectors.npz", test_vectors)

    print("Save test labels...")
    np.save("./data/test_labels.npy", test_labels)
