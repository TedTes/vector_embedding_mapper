import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

# Load pre-trained GloVe model (50-dim for speed; downloads ~70MB first time)
print("Loading GloVe model...")
model = api.load("glove-wiki-gigaword-50")
print("Model loaded!")

def get_embedding(word):
    """Get normalized vector for a word."""
    try:
        vec = model[word.lower()]
        return vec / np.linalg.norm(vec)
    except KeyError:
        print(f"Word '{word}' not in vocabulary.")
        return None

def analogy(a, b, c):
    """Compute analogy: a - b + c ≈ ?"""
    vec_a = get_embedding(a)
    vec_b = get_embedding(b)
    vec_c = get_embedding(c)
    
    if vec_a is None or vec_b is None or vec_c is None:
        return None
    
    result_vec = vec_a - vec_b + vec_c
    result_vec /= np.linalg.norm(result_vec)
    similar = model.similar_by_vector(result_vec, topn=5)
    print(f"\nAnalogy: {a} - {b} + {c} ≈ ?")
    for word, score in similar:
        print(f"  {word}: {score:.3f}")
    return similar[0][0]

def plot_vectors(words, title="Embeddings in 2D", dim=2):
    """Project to 2D/3D and plot."""
    vectors = [get_embedding(w) for w in words if get_embedding(w) is not None]
    if len(vectors) < 2:
        print("Not enough valid words.")
        return
    vectors = np.array(vectors)
    pca = PCA(n_components=dim)
    projected = pca.fit_transform(vectors)
    
    if dim == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(projected[:, 0], projected[:, 1])
        for i, word in enumerate(words):
            plt.annotate(word, (projected[i, 0], projected[i, 1]))
        plt.title(title)
        plt.grid(True)
        plt.show()
    elif dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2])
        for i, word in enumerate(words):
            ax.text(projected[i, 0], projected[i, 1], projected[i, 2], word)
        ax.set_title(title)
        plt.show()

# Demo
analogy("king", "man", "woman")
analogy("paris", "france", "italy")

plot_words = ["king", "queen", "man", "woman", "paris", "france", "rome", "italy"]
plot_vectors(plot_words, dim=2)  # Or dim=3 for 3D