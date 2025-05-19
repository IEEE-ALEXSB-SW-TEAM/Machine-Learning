from collections import defaultdict, Counter

def get_vocab(corpus):
    """Splits words into characters with an end-of-word marker."""
    vocab = defaultdict(int)
    for word in corpus.strip().split():
        chars = " ".join(list(word)) + " </w>"  # End-of-word token
        vocab[chars] += 1
    return vocab

def get_stats(vocab):
    """Count frequency of character pairs in the vocab."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the most frequent pair in vocab."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

corpus = "Hello I am a Here"
vocab = get_vocab(corpus)

print("Initial vocab:")
for word, freq in vocab.items():
    print(f"{word} -> {freq}")

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    print(f"\nStep {i+1}: merging {best_pair} (freq: {pairs[best_pair]})")
    vocab = merge_vocab(best_pair, vocab)
    for word, freq in vocab.items():
        print(f"{word} -> {freq}")
