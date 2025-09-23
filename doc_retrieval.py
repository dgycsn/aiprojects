import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import PunktTokenizer
import faiss


curr_user = os.environ.get('USERNAME')
mydir = "C:/Users/" + curr_user + "/Downloads/"
# mydir = "C:\\"
os.chdir(mydir)

docname = "ecm.pdf"

# %%
# import pdf
reader = PdfReader(docname)

# # printing number of pages in pdf file
# print(len(reader.pages))

# # creating a page object
# page = reader.pages[0]

# # extracting text from page
# print(page.extract_text())

fulldoc = ""
for page in reader.pages:
    mytext = page.extract_text()
    fulldoc += mytext + "\n"


# %%
# split pdf into sentences
# see also: https://www.nltk.org/api/nltk.tokenize.punkt.html

tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
# sent_detector = PunktTokenizer()

splitdoc = tokenizer.tokenize(fulldoc.strip())
# print('\n-----\n'.join(splitdoc))

# %%

model_path = mydir + "multi-qa-mpnet-base-dot-v1"
model = SentenceTransformer(model_path)

doc_embeddings = model.encode(splitdoc)

# %%

# Dimension of embeddings
d = doc_embeddings.shape[1]

# Build the index (cosine similarity ~ inner product with normalized vectors)
index = faiss.IndexFlatIP(d)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(doc_embeddings)

# Add documents to index
index.add(doc_embeddings)

# %%

# Example query
query = "was ist fabasoft"
query_embedding = model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(query_embedding)

# Search top-k
k = 5
distances, indices = index.search(query_embedding, k)

print("\nQuery:", query, "\n")

for idx, score in zip(indices[0], distances[0]):
    print(f"{splitdoc[idx]} \n(score: {score:.4f}) \n ----")
