import json
from bareunpy import Tagger, Tokenizer
from google.protobuf.json_format import MessageToDict
import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
import tensorflow as tf
import os
import pdb
import re
# 예제 문장
txt_files = os.listdir('data/')

sentences = []
for text_file in txt_files:
    # 각 파일을 열고 모든 줄을 읽습니다.
    with open(f'data/{text_file}', 'r', encoding='utf-8') as file:
        # 파일의 모든 줄을 sentences 리스트에 추가합니다.
        sentences.extend(file.readlines())

tagger = Tagger(apikey='2AVZBLY-D7QUNYQ-XNCRIOA-NVP3TLA', host='localhost')
tokenizer = Tokenizer(apikey='2AVZBLY-D7QUNYQ-XNCRIOA-NVP3TLA', host='localhost')
# 형태소 분석을 통한 토큰화

tokenized = tokenizer.tokenize_list(sentences)
tokenized = MessageToDict(tokenized.r)


# 조사를 제거하는 함수
def remove_particles(sentence):
    new_tokens = []
    
    for token in sentence['tokens']:
        for seg in token['segments']:
            token_tag= f"{seg['text']['content']}/{seg['hint']}"
            if token_tag not in ['/J', '/E']:
                new_tokens.append(token_tag)
    return new_tokens

# 각 문장에 대해 조사 제거 실행
sentences = []
for sentence in tokenized['sentences']:
    no_particles = remove_particles(sentence)
    sentences.append(no_particles)

# Word2Vec 모델 학습
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)


# 임베딩 벡터와 메타데이터 파일 저장
embeddings = model.wv.vectors
words = model.wv.index_to_key

# TensorBoard 로깅 경로 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 임베딩 벡터 저장
embeddings_tensor = tf.Variable(embeddings, name="word_embeddings")
checkpoint = tf.train.Checkpoint(embedding=embeddings_tensor)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# 메타데이터(단어 목록) 파일 저장
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for word in words:
        f.write(f"{word}\n")

# 프로젝터 설정
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# Word2Vec 모델로부터 임베딩 벡터 가져오기
embeddings = model.wv.vectors
word_labels = model.wv.index_to_key

# t-SNE 모델 생성 및 임베딩 벡터 변환
tsne_model = TSNE(n_components=2, perplexity=30,random_state=0)
reduced_embeddings = tsne_model.fit_transform(embeddings)

# 시각화
plt.figure(figsize=(10, 10))
for i, label in enumerate(word_labels):
    x, y = reduced_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()
print('a')