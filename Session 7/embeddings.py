import gensim.downloader as api

model = api.load("glove-twitter-25")

target_word = 'king'

if target_word in model:
    print(f"Top 5 words similar to '{target_word}':")
    for word, similarity in model.most_similar(target_word, topn=5):
        print(f"{word} ({similarity:.3f})")
else:
    print(f"'{target_word}' not in vocabulary")

print('-----')

print(f"Most similar to {target_word}:")
for word, sim in model.most_similar(target_word, topn=5):
    print(f"{word}: {sim:.3f}")

# Analogy: king - man + woman = ?
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)
print("\nAnalogy: king - man + woman = ", result)
