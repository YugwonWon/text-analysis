import json
import glob
import csv
import os
from tqdm import tqdm

from bareunpy import Tagger

import pyLDAvis.gensim_models
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from google.protobuf.json_format import MessageToDict

# 형태소 분석을 통해 명사만 추출하는 함수
def extract_nouns(text):
    nouns = []
    analysis_results = tagger.tag(text)
    for sentence in analysis_results.sentences:
        for token in sentence.tokens:
            if token.part_of_speech.startswith('NNG') and token.text not in TopicModeling.stopwords:  # 명사(Noun)인 경우만 선택
                nouns.append(token.text)
    return nouns


class TopicModeling:
    stopwords = set(['들', '그', '저'])
    
    def __init__(self, documents, tagger):
        self.documents = documents
        self.tagger = tagger
        
    def make_join_sentence(self, documents):
        len_lines = len(documents)
        num_iters = len_lines // 2 + (1 if len_lines % 2 != 0 else 0)
        join_texts = []
        for i in range(num_iters):
            joined_text = " ".join(documents[2 * i: min(2 * (i + 1), len_lines)])
            join_texts.append(joined_text)
        return join_texts

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
                        morph["text"]["content"] not in TopicModeling.stopwords:  # 명사를 확인하는 조건
                        nouns.append(morph["text"]["content"])  # 명사만을 리스트에 추가
        return nouns

    def preprocess_documents(self):
        processed_docs_by_code = {}
        for code, docs in tqdm(self.documents.items(), desc='documents'):
            # join_texts = self.make_join_sentence(docs)
            processed_documents = []
            for text in tqdm(docs, desc='category'):
                nouns = self.extract_nouns(text)
                if len(nouns)==0:
                    print('Error AnalyzeText')
                    continue
                processed_documents.extend(nouns)
            processed_docs_by_code[code] = processed_documents
        return processed_docs_by_code

    def run_lda(self, processed_docs_by_code):
        lda_models_by_code = {}
        for code, docs in processed_docs_by_code.items():
            dictionary = corpora.Dictionary([docs])
            corpus = [dictionary.doc2bow(doc) for doc in [docs]]
            lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=15)
            lda_models_by_code[code] = [lda_model, corpus]
        return lda_models_by_code

    def run(self, num_topics=11, num_words=5, output_dir='out/csv'):
        code_dict = {}
        processed_docs_by_code = self.preprocess_documents()
        lda_models_by_code = self.run_lda(processed_docs_by_code)
        # 각 분야별 토픽 출력
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'{output_dir}/LDA분석_결과.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Code', 'Topic Index', 'Topic'])
            for code, [lda_model, corpus] in lda_models_by_code.items():
                print(f"Code: {code}")
                dictionary = lda_model.id2word  # 수정된 run_lda 메서드로부터 반환받은 값
                for idx, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=num_words):
                    print(f"Topic {idx}: {topic}")
                    writer.writerow([code, f"Topic {idx}", topic])
                print("\n")
                code_dict[code] = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        return code_dict

if __name__ == '__main__':    
    # 바른 형태소 분석기 초기화
    tagger = Tagger("koba-6BVW24Q-2MIEHVQ-WQAVXJI-E2TRS7A", 'localhost', 5656)
    
    # 뉴스 데이터 불러오기 및 전처리
    news_data = []
    for filename in glob.glob('/home/yugwon/text-analysis/data/news/01.데이터/1.Training/**/*.json', recursive=True):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for article in data["data"]:
                code = article["doc_class"]["code"]
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    news_data.append((code, context))
                    
    # code 별로 문장을 분류
    news_by_code = {}
    for code, context in news_data:
        if code not in news_by_code:
            news_by_code[code] = []
        news_by_code[code].append(context)
    
    # _news_by_code = {}
    # for code, context in news_by_code.items():
    #     _news_by_code[code] = context[:5]
    # news_by_code = _news_by_code
        
    tm = TopicModeling(documents=news_by_code, tagger=tagger)
    code_dict = tm.run(num_topics=5, num_words=5)
    import pickle
    with open('code_dict.pkl', 'wb') as file:
        pickle.dump(code_dict, file)
    pyLDAvis.enable_notebook()
    # 시각화
    pyLDAvis.display(code_dict['정치'])
    
    
