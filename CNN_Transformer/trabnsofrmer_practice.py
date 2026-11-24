import torch
from torch import nn
words=['we','study','python','we','study','torch']
unique_words=sorted(set(words))
print(unique_words)

vocabulary={}
for i,word in enumerate(unique_words):
    vocabulary[word]=i
first_id=[]
for word in words[:4]:
    first_id.append(vocabulary[word])
token_id=torch.Tensor([first_id])
print(token_id)
embedding=nn.Embedding(num_embeddings=len(vocabulary),embedding_dim=3)
word_vectors=embedding(token_id)

print(word_vectors.shape)