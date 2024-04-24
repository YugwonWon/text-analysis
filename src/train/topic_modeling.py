import json
import glob
import csv
import os
import pandas as pd
from tqdm import tqdm

from bareunpy import Tagger

import pyLDAvis.gensim_models
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from google.protobuf.json_format import MessageToDict

# 형태소 분석을 통해 명사만 추출하는 함수
def extract_nouns(text):
    """
    형태소 분석을 통해 명사만을 추출합니다.
    :param text(str): 분석할 텍스트
    """
    nouns = []
    analysis_results = tagger.tag(text)
    for sentence in analysis_results.sentences:
        for token in sentence.tokens:
            if token.part_of_speech.startswith('NNG') and token.text not in TopicModeling.stopwords:  # 명사(Noun)인 경우만 선택
                nouns.append(token.text)
    return nouns

def load_data(path):
    """
    폴더에서 json 목록을 구해 원하는 텍스트를 추출합니다.
    :param path: json이 들어 있는 파일 경로
    """
    print('> 데이터 로딩')
    json_list = glob.glob(os.path.join(path, '**/*.json'), recursive=True)
    # 뉴스 데이터 불러오기
    news_data = []
    for filename in json_list:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for article in data["data"]:
                code = article["doc_class"]["code"] # 카테고리
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"] # 뉴스기사 본문
                    news_data.append((code, context))
    print('> 데이터 로딩 완료\n')
    return news_data


class TopicModeling:
    stopwords = set(['들']) # 불용어가 필요한 경우 여기에 추가합니다.
    
    def __init__(self, documents, tagger):
        self.documents = documents
        self.tagger = tagger

    def extract_nouns(self, text):
        try:
            res = self.tagger.tag(text)  # 형태소 분석 호출
        except:
            return []
        res = MessageToDict(res.r)
        nouns = []
        for sentence in res["sentences"]:
            for token in sentence["tokens"]:
                for morph in token["morphemes"]:
                    if morph["tag"].startswith('NNG') and \
                        morph["text"]["content"] not in TopicModeling.stopwords:  # 명사를 확인하는 조건: 태그가 명사이고, 불용어 리스트에 없는 경우
                        nouns.append(morph["text"]["content"])  # 명사만을 리스트에 추가
        return nouns

    def preprocess_documents(self):
        print('> 전처리 실행(형태소 분석, 명사추출)')
        processed_documents = []
        for _, text in tqdm(self.documents, desc='documents'):
            nouns = self.extract_nouns(text)
            if len(nouns)==0:
                print('Error AnalyzeText')
                continue
            processed_documents.append(nouns)
        print('> 전처리 완료\n')
        print('> 전처리된 뉴스 데이터 샘플')
        for doc in processed_documents[:5]:  # 처음 5개 문서만 출력
          print(' '.join(doc))
        return processed_documents

    def run_lda_on_combined_corpus(self, processed_docs):
        print('\n> LDA 학습 시작')
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=15)
        print('> LDA 학습 종료')
        return lda_model, corpus, dictionary

    def run(self, num_topics=10, num_words=5, output_dir='out/csv'):
        print('\n> 토픽 모델링 시작')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 전처리
        processed_docs = self.preprocess_documents()

        #LDA 학습
        lda_model, corpus, dictionary = self.run_lda_on_combined_corpus(processed_docs)

        # LDA 결과 저장
        with open(f'{output_dir}/LDA분석_결과.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Topic Index', 'Topic'])
            for idx, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=num_words):
                writer.writerow([f"Topic {idx}", topic])

        # LDAvis 준비 및 반환
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        print('> LDAvis 생성')
        return vis_data

if __name__ == '__main__':
    # 바른 형태소 분석기 초기화
    with open('config.json', 'r') as f:
        config = json.load(f)
    tagger = Tagger(config['bareun_api_key'], 'localhost', 5757)

    news_data = load_data(path="data/news")
    news_df = pd.DataFrame(news_data)
    print('뉴스 데이터 샘플')
    print(news_df.head(5))

    # 테스트를 위해 500개만 선택(전체 샘플은 약 16만 문장)
    news_data = news_data[:500]

    # 토픽 모델링을 시작합니다.
    tm = TopicModeling(documents=news_data, tagger=tagger)
    vis_data = tm.run(num_topics=10, num_words=5)

    # 디렉토리가 없다면 만듭니다.
    if not os.path.exists('out/pkl'):
        os.makedirs('out/pkl')

    import pickle
    # 저장할 파일명 수정
    save_name = 'out/pkl/vis_data2.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(vis_data, file)
    print(f'> LDAvis data 저장 완료 -> {save_name}')
    # pyLDAvis.enable_notebook()
    # # 시각화
    # pyLDAvis.display(vis_data)
    
    
