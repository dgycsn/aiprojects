from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = "C:\\Users\\alp.yuecesan\\Downloads\\all-MiniLM-L6-v2"

model = SentenceTransformer(model_path)

# %%

# sentences = ["This is an example sentence", "Each sentence is converted"]
# sentences = "hello"
sentences = "good"
embeddings1 = model.encode(sentences)
print((max(embeddings1), np.argmax(embeddings1), np.linalg.norm(embeddings1)))

sentences = "great"
embeddings2 = model.encode(sentences)
print((max(embeddings2), np.argmax(embeddings2), np.linalg.norm(embeddings2)))


sentences = "bad"
embeddings3 = model.encode(sentences)
print((max(embeddings3), np.argmax(embeddings3), np.linalg.norm(embeddings3)))

print("euclidian norm")
print(np.linalg.norm(embeddings1 - embeddings2))
print(np.linalg.norm(embeddings1 - embeddings3))

print("cosine similarity")
print(util.cos_sim(embeddings1, embeddings2))
print(util.cos_sim(embeddings1, embeddings3))

norm = np.linalg.norm(embeddings1 - embeddings2)
cosim = util.cos_sim(embeddings1, embeddings2)

# %%
# since vectors are normalized, cosine and euclidian should deliver the same result
# convert cosine similarity to euclidian norm to check

angle = np.arccos(cosim)
# since both vectors are on unit circle, can calculate length with the perpendicular triangles twice
length = np.sin(angle/2) * 2
print("cosine sim = " + str(cosim))
print("norm = " + str(norm))
print("calculated norm = " + str(length))

# %%

sentences = "good"
embeddings1 = model.encode(sentences)
print((max(embeddings1), np.argmax(embeddings1), np.linalg.norm(embeddings1)))

sentences = "best"
embeddings2 = model.encode(sentences)
print((max(embeddings2), np.argmax(embeddings2), np.linalg.norm(embeddings2)))


sentences = "worst"
embeddings3 = model.encode(sentences)
print((max(embeddings3), np.argmax(embeddings3), np.linalg.norm(embeddings3)))

print(np.linalg.norm(embeddings1 - embeddings2))
print(np.linalg.norm(embeddings1 - embeddings3))

print(util.cos_sim(embeddings1, embeddings2))
print(util.cos_sim(embeddings1, embeddings3))

# %%
"""
1 good vs great & bad

Euclidean distance:
good–great: 0.74 (closer)
good–bad: 0.91 (farther)

Cosine similarity
good–great: 0.73 (pretty high)
good–bad: 0.59 (moderate, but still related)

As expected: good is closer to great than bad, but bad isn’t "far away" because antonyms share context.

2 good vs best & worst

Euclidean distance
good–best: 1.01
good–worst: 1.09

Cosine similarity
good–best: 0.48
good–worst: 0.40

best is not that close to good, even though they’re semantically related.
worst is a bit further away, which fits our intuition.

Embeddings capture “usage similarity,” not strict dictionary relations.
MiniLM is optimized for general semantic similarity. --> It may not capture fine-grained word hierarchy (good < better < best) as strongly.

--> Embeddings =/= dictionary synonyms.

"""
# %%

# another way of calculating the comparison
query = "good"
docs = ["great", "bad"]


# Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)

# Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

# Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

# Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

# Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)

# %%
sentences = "gut"
embeddings1 = model.encode(sentences)

sentences = "schön"
embeddings2 = model.encode(sentences)


sentences = "schlecht"
embeddings3 = model.encode(sentences)

print(np.linalg.norm(embeddings1 - embeddings2))
print(np.linalg.norm(embeddings1 - embeddings3))

print(util.cos_sim(embeddings1, embeddings2))
print(util.cos_sim(embeddings1, embeddings3))
