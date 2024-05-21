import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from google.protobuf.json_format import MessageToDict
from bareunpy import Tagger

class NewsRecommendation:
    def __init__(self, csv_file_path):
        self.tagger = Tagger("your-api-key", 'localhost', port=5757)
        self.stopwords = ['들']
        self.data = pd.read_csv(csv_file_path)  # CSV 파일 불러오기
        self.process_data()
        
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
                    if (morph["tag"].startswith('NNG') or morph["tag"].startswith('NNP')) and \
                        morph["text"]["content"] not in self.stopwords:  # 명사를 확인하는 조건: 태그가 명사이고, 불용어 리스트에 없는 경우
                        nouns.append(morph["text"]["content"])  # 명사만을 리스트에 추가
        return nouns

    def process_data(self):
        # 기사 제목에서 명사 추출
        self.data['noun'] = self.data['title'].apply(lambda x: self.extract_nouns(x))
        self.data['noun_str'] = self.data['noun'].apply(lambda x: ' '.join(x))
        
        # 'hit' 열을 숫자형으로 변환
        self.data['hit'] = pd.to_numeric(self.data['hit'], errors='coerce')
        
        # NaN 값 제거
        self.data = self.data.dropna(subset=['hit'])
        
        # 조회수의 로그 값 계산
        self.data['hit(log)'] = np.log(self.data['hit'])
        
        # 명사 가중 빈도수 및 전체 빈도수 계산
        self.word_freq = {}
        self.word_count = {}
        for idx, row in self.data.iterrows():
            weight = row['hit(log)']
            for noun in row['noun']:
                if noun in self.word_freq:
                    self.word_freq[noun] += weight
                    self.word_count[noun] += 1
                else:
                    self.word_freq[noun] = weight
                    self.word_count[noun] = 1

        # DataFrame으로 변환
        self.word_freq_df = pd.DataFrame(list(self.word_freq.items()), columns=['word', 'weighted_frequency'])
        self.word_count_df = pd.DataFrame(list(self.word_count.items()), columns=['word', 'frequency'])
        
        # 가중 빈도수 및 전체 빈도수 데이터프레임 병합
        self.word_stats_df = pd.merge(self.word_freq_df, self.word_count_df, on='word')
        
        # 가중 빈도수로 정렬
        self.word_stats_df = self.word_stats_df.sort_values(by='weighted_frequency', ascending=False).reset_index(drop=True)
        
        # 단어별 가중치 및 빈도 데이터를 CSV 파일로 저장 (내림차순으로 정렬)
        self.word_stats_df.to_csv('word_weighted_frequency.csv', index=False, encoding='utf-8-sig')
        
        # 전체 데이터프레임도 CSV 파일로 저장
        self.data.to_csv('processed_articles.csv', index=False, encoding='utf-8-sig')

    def recommend_related_articles(self, user_title, top_n=5):
        # 사용자 입력 뉴스 기사 제목에서 명사 추출
        user_nouns = self.extract_nouns(user_title)
        user_nouns_str = ' '.join(user_nouns)
        
        # CountVectorizer를 사용하여 벡터화
        vectorizer = CountVectorizer().fit(self.data['noun_str'])
        user_vector = vectorizer.transform([user_nouns_str])
        data_vectors = vectorizer.transform(self.data['noun_str'])
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(user_vector, data_vectors)[0]
        
        # 명사별 가중치 계산
        weighted_similarity = []
        for idx, row in self.data.iterrows():
            weight_sum = 0
            for noun in user_nouns:
                if noun in row['noun']:
                    weight_sum += self.word_stats_df[self.word_stats_df['word'] == noun]['weighted_frequency'].values[0]
            weight_avg = weight_sum / len(user_nouns) if user_nouns else 0  # 명사의 평균 가중치 계산
            weighted_similarity.append(similarity[idx] * weight_avg)
        
        self.data['weighted_similarity'] = weighted_similarity
        
        # 가중 유사도 기준으로 정렬
        recommended_articles = self.data.sort_values(by='weighted_similarity', ascending=False).head(top_n)
        
        # 추천 기사 데이터를 CSV 파일로 저장
        recommended_articles.to_csv('recommended_related_articles.csv', index=False, encoding='utf-8-sig')
        
        return recommended_articles[['title', 'hit', 'weighted_similarity']]

# CSV 파일 경로를 지정하여 NewsRecommendation 클래스 초기화
csv_file_path = '국민일보기사_조회수.csv'
word_stats_file_path = 'word_weighted_frequency.csv'
news_recommendation = NewsRecommendation(csv_file_path)

# 사용자가 선택한 뉴스 기사 제목
user_title = "김성태, ‘檢 술자리 회유’ 반박 “이화영, 검찰 검사 앞에서 탁자 치고 소리쳐”"

# 관련 기사 추천
recommended_related_articles = news_recommendation.recommend_related_articles(user_title)
print("\n추천 관련 기사:")
print(recommended_related_articles)

# 단어별 가중치 및 빈도 데이터 출력
print("\n단어별 가중치 및 빈도 데이터:")
print(news_recommendation.word_stats_df)
