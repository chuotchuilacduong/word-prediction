import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 8

from platform import python_version
assert int(python_version().split(".")[1]) >= 5
from gensim.models import KeyedVectors
import gensim.downloader as api
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
from datasets import load_dataset
import re
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'
NUM_SAMPLES = 150
np.random.seed(0)
random.seed(0)

# Load IMDb dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")

def read_corpus():
    """ Read files from the Large Movie Review Dataset. """
    files = imdb_dataset["train"]["text"][:NUM_SAMPLES]
    return [[START_TOKEN] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus. """
    flat = [word for sublist in corpus for word in sublist]
    unique = set(flat)
    corpus_words = sorted(unique)
    n_corpus_words = len(corpus_words)
    return corpus_words, n_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size. """
    words, n_words = distinct_words(corpus)
    M = np.zeros((n_words, n_words))
    word2ind = {word: i for i, word in enumerate(words)}
    for document in corpus:
        for i, word in enumerate(document):
            word_index = word2ind[word]
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word = document[j]
                    context_index = word2ind[context_word]
                    M[word_index, context_index] += 1
    return M, word2ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality to k using TruncatedSVD. """
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    svd = TruncatedSVD(n_components=k)
    M_reduced = svd.fit_transform(M)
    print("Done.")
    return M_reduced

def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words". """
    plt.figure(figsize=(10, 8))
    indices = [word2ind[word] for word in words]
    plt.scatter(M_reduced[indices, 0], M_reduced[indices, 1], color='red')
    for i, word in enumerate(words):
        plt.text(M_reduced[word2ind[word], 0], M_reduced[word2ind[word], 1], word, fontsize=12, ha='right', color='black')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Visualization of Word Embedding')
    plt.grid(True)
    plt.show()

def load_embedding_model():
    """ Load GloVe Vectors """
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin

def analyze_word(wv_from_bin, word):
    """ Analyze the top-10 most similar words for a given word. """
    try:
        top_similar_words = wv_from_bin.most_similar(word, topn=10)
        return top_similar_words
    except KeyError:
        print(f"The word '{word}' is not in the vocabulary.")
        return []

def find_cosine_distances(wv_from_bin, synonyms, antonyms):
    """
    Find words where the cosine distance between synonyms is greater than that between antonyms.
    """
    for syn_pair in synonyms:
        for ant_pair in antonyms:
            # Compute cosine distances
            syn_distance = wv_from_bin.distance(syn_pair[0], syn_pair[1])
            ant_distance = wv_from_bin.distance(ant_pair[0], ant_pair[1])
            
            # Check the condition
            if syn_distance > ant_distance:
                return syn_pair, ant_pair, syn_distance, ant_distance
    return None, None, None, None

def find_analogy(wv_from_bin, x, y, a):
    """
    Find the word b such that x : y :: a : b according to the word vectors.
    """
    result = wv_from_bin.most_similar(positive=[a, y], negative=[x], topn=1)
    b = result[0][0]
    return b

def find_incorrect_analogy(wv_from_bin, x, y, a, expected_b):
    """
    Find a word b that completes the analogy x : y :: a : b, and check if it does not match the expected_b.
    """
    result = wv_from_bin.most_similar(positive=[a, y], negative=[x])
    incorrect_b = result[0][0]
    print(f"Intended analogy: {x} : {y} :: {a} : {expected_b}")
    print(f"Computed analogy: {x} : {y} :: {a} : {incorrect_b}")
    return incorrect_b

def find_biased_analogy(wv_from_bin, x, y, a):
    """
    Find the most similar word to complete the analogy x : y :: a : b and check for biases.
    """
    result = wv_from_bin.most_similar(positive=[a, y], negative=[x])
    b = result[0][0]
    print(f"Analogy: {x} : {y} :: {a} : {b}")
    return b

# Main function to find the analogy
def main():
    # Load embeddings
    wv_from_bin = load_embedding_model()
    
    # Input from user
    x = 'cat'
    y = 'dog'
    a = 'machine'
    
    print("Finding analogy...")
    b = find_analogy(wv_from_bin, x, y, a)
    
    print(f"Analogy result: {x} : {y} :: {a} : {b}")

if __name__ == "__main__":
    main()
