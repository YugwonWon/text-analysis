import os
import re
import json
import gensim
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorboard.plugins import projector
from bareunpy import Tagger, Tokenizer
from google.protobuf.json_format import MessageToDict

def read_text_files(directory):
    """ 
    지정된 디렉토리에서 텍스트 파일을 읽어 각 파일의 모든 줄을 리스트로 반환합니다.
    :param directory: 텍스트 파일이 저장된 디렉토리 경로
    :return: 파일의 모든 줄을 포함하는 리스트
    """
    sentences = []
    for text_file in os.listdir(directory):
        with open(f'{directory}/{text_file}', 'r', encoding='utf-8') as file:
            sentences.extend(file.readlines())
    return sentences

def initialize_bareunpy():
    """ 
    설정 파일을 읽고 BareunPy Tagger 및 Tokenizer 객체를 초기화합니다.
    :return: 초기화된 Tokenizer 객체
    """
    with open('config.json', 'r') as f:
        config = json.load(f)
    tagger = Tagger(apikey=config['bareun-api-key'], host='localhost')
    tokenizer = Tokenizer(apikey=config['bareun-api-key'], host='localhost')
    return tokenizer

def tokenize_sentences(tokenizer, sentences):
    """ 
    주어진 문장 리스트에 대해 형태소 분석을 수행하고 결과를 Python 딕셔너리로 변환합니다.
    :param tokenizer: 형태소 분석을 수행할 Tokenizer 객체
    :param sentences: 형태소 분석을 수행할 문장의 리스트
    :return: Python 딕셔너리 형태로 변환된 토큰화된 문장 데이터
    """
    tokenized = tokenizer.tokenize_list(sentences)
    return MessageToDict(tokenized.r)  # protobuf 객체를 딕셔너리로 변환

def remove_particles(tokenized):
    """ 
    토큰화된 문장에서 조사를 제거합니다.
    :param tokenized: 토큰화된 문장 데이터
    :return: 조사가 제거된 문장 리스트
    """
    def check_pattern(token_tag):
        pattern = r'/(J|E|S|E)$|\d+/N|^[가-핳]{1}/(A|N)$|^(하|이|되|있)/V$'
        return not re.search(pattern, token_tag)

    sentences = []
    for sentence in tokenized['sentences']:
        new_tokens = []
        for token in sentence['tokens']:
            for seg in token['segments']:
                token_tag = f"{seg['text']['content']}/{seg['hint']}"
                if check_pattern(token_tag):
                    new_tokens.append(token_tag)
        sentences.append(new_tokens)
    return sentences

def train_word2vec_model(sentences):
    """ 
    주어진 문장으로 Word2Vec 모델을 학습합니다.
    :param sentences: 학습에 사용할 문장 리스트
    :return: 학습된 Word2Vec 모델
    """
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def setup_tensorboard_embeddings(model, log_dir='logs'):
    """ 
    Word2Vec 모델의 임베딩을 TensorBoard에 로깅합니다.
    :param model: Word2Vec 모델
    :param log_dir: TensorBoard 로깅을 저장할 디렉토리
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    embeddings_tensor = tf.Variable(model.wv.vectors, name="word_embeddings")
    checkpoint = tf.train.Checkpoint(embedding=embeddings_tensor)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        for word in model.wv.index_to_key:
            f.write(f"{word}\n")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)

def visualize_tsne(embeddings, labels):
    """ 
    t-SNE를 사용하여 임베딩 벡터를 2차원으로 축소하고 시각화합니다.
    :param embeddings: 임베딩 벡터
    :param labels: 각 벡터에 해당하는 레이블(단어)
    """
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=0)
    reduced_embeddings = tsne_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


if __name__ == '__main__':
    # 메인 코드 실행
    sentences = read_text_files('data/word2vec')
    tokenizer = initialize_bareunpy()
    tokenized = tokenize_sentences(tokenizer, sentences)
    sentences_no_particles = remove_particles(tokenized)
    model = train_word2vec_model(sentences_no_particles)
    setup_tensorboard_embeddings(model)
    visualize_tsne(model.wv.vectors, model.wv.index_to_key)