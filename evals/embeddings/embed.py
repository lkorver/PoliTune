from transformers import AutoTokenizer, AutoModel
import torch
from torchtune import config

from gensim.models.doc2vec import Doc2Vec,\
	TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import argparse

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Right or left.",
    required=True
)
parser.add_argument(
    "--outfile",
    type=str,
    help="The output csv file.",
    required=True
)
parser.add_argument(
    "--infile",
    type=str,
    help="The csv file containing the data",
    required=True
)
parser.add_argument(
    "--infile2",
    type=str,
    help="The csv file containing the data",
    required=False
)

test_cases = ["Abortion", "Healthcare", "Carbon Tax"]
count = 3

args = parser.parse_args()


r_data = []
r_iter_step = []

with open(args.infile, 'r', newline='') as f:
    reader = csv.reader(f)
    for row_index, row in enumerate(reader):
        if row_index != 0:
            r_data.append([row[2],row[3],row[4]])
            # print(row[3])
            r_iter_step.append([row[0],row[1]])

l_data = []
l_iter_step = []
with open(args.infile2, 'r', newline='') as f:
    reader = csv.reader(f)
    for row_index, row in enumerate(reader):
        if row_index != 0:
            l_data.append([row[2],row[3],row[4]])
            # print(row[3])
            l_iter_step.append([row[0],row[1]])

csv_headers = ['iteration','step'] 
for i in range(count):
    csv_headers.append(f'prompt_{i}')
# print("headers: ",csv_headers)

with open(args.outfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)


# preproces the documents, and create TaggedDocuments
tagged_data = []
all_data = []
ground_labels = []
for i, doc_dict in enumerate(r_data):
    for j in range(3):
        all_data.append(doc_dict)
        tagged_data.append(TaggedDocument(words=word_tokenize(doc_dict[j]), tags=["Right",test_cases[j]]))
        ground_labels.append(f"Right {test_cases[j]}")
for i, doc_dict in enumerate(l_data):
    for j in range(3):
        all_data.append(doc_dict)
        tagged_data.append(TaggedDocument(words=word_tokenize(doc_dict[j]), tags=["Left",test_cases[j]]))
        ground_labels.append(f"Left {test_cases[j]}")

# train the Doc2vec model
model = Doc2Vec(vector_size=100,
                min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

# get the document vectors
for i, doc_dict in enumerate(l_data):
    # print(doc_dict)
    document_vectors = [model.infer_vector(
        word_tokenize(doc)) for doc in doc_dict]

    # # save to outfile 
    # with open(args.outfile, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(l_iter_step[i]+document_vectors)

model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(all_data)
similarities = model.similarity(embeddings, embeddings)
print(similarities[0])

kmeans = KMeans(n_clusters=2)  
kmeans.fit(embeddings)
labels = kmeans.labels_

count = 0
for doc, label in zip(all_data, labels):
    # print(f"data: {doc}, Cluster: {label}")
    print(f" Cluster: {label}, ground: {ground_labels[count]}")
    count += 1


