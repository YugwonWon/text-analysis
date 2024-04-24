import json
import os
import glob
import csv
import re
import logging

# 로거 생성 및 로그 레벨 설정
logger = logging.getLogger('custom_dict_example')
logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, CRITICAL 중 원하는 레벨 설정

# 파일 핸들러 생성 및 로그 파일 설정
file_handler = logging.FileHandler('out/analyzer.log')
file_handler.setLevel(logging.DEBUG)  # 파일에 기록될 로그 레벨

from bareunpy import Tagger
from google.protobuf.json_format import MessageToDict
from wordcloud import WordCloud
from collections import Counter

class CustomDictAnalyzer:
    """
    바른 형태소 분석기의 사용자정의 사전 기능을 활용해 텍스트 분석을 수행합니다.
    """
    def __init__(self, corpus_list):
        """
        코퍼스 디렉토리에서 json파일 및 wav 파일 목록을 생성합니다.
        :param corpus_list: 여러 코퍼스 리스트를 입력으로 받습니다.
        """
        # bareun api-key를 불러옵니다.
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # bareun 형태소 분석 객체를 불러옵니다.
        self.tagger = Tagger(self.config['bareun_api_key'], 'bigkinds-apiLB-2834425-f49051393917.kr-gov.lb.naverncp.com', port=5757)

        # json파일 및 wav파일 목록을 만듭니다.
        self.corpus_list = corpus_list
        self.corpus_json_dict = {}
        for idx, corpus in enumerate(corpus_list):
            json_files = glob.glob(os.path.join(corpus, '**/*.json'), recursive=True)
            self.corpus_json_dict[f'{str(idx)}_json'] = json_files # 요청된 corpus 리스트 순서대로 번호를 부여한다.

        self.results = []

    def mapping_name(self, target_dir_files):
        """
        파일 경로의 목록을 받아 파일 이름을 key로, 전체 파일 경로를 값으로 하는 사전을 생성합니다.
        :param target_dir_files: 분석 대상 디렉토리 내의 파일 목록
        :return wav_id_dict: 파일 이름을 기준으로 한 파일 경로의 사전(dictionary)
        """
        wav_id_dict = {}
        for wav in target_dir_files:
            bn =  os.path.basename(wav).split('.')[0]
            wav_id_dict[bn] = wav

        return wav_id_dict

    def extract_word_sent(self, corpus_id, out_name):
        """
        단어와 문장을 파일로 추출합니다.
        :param corpus_id: 코퍼스 번호
        :param out_name: 저장파일 이름
        :return: json, txt 파일
        """
        if corpus_id == 0:
            self.total_sents = []
            self.total_words = []
            for file in self.corpus_json_dict[f'{str(corpus_id)}_json']:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                word_info_dict = {}
                for dialog in data["Dialogs"]:
                    self.total_sents.append(dialog["SpeakerText"]) # 전체 문장 저장
                    if dialog["WordInfo"] is None:
                        continue
                    for word_info in dialog["WordInfo"]:
                        word = word_info["Word"]
                        word_info_dict[word] = {
                            "WordType": word_info["WordType"],
                            "WordStructure": word_info["WordStructure"],
                            "WordDefine": word_info["WordDefine"]
                        }

                self.total_words.append(word_info_dict) # 전체 단어 저장

        elif corpus_id == 1:
            # 코퍼스 json구조에 맞게 수정합니다.
            pass

        if not os.path.isdir('out/json/corpus'):
            os.makedirs('out/json/corpus')
        with open(f'out/json/corpus/{out_name}.json', 'w', encoding='utf-8') as file:
            json.dump(self.total_words, file, ensure_ascii=False, indent=4)
        print(f'단어 파일 저장 완료 --> out/json/corpus/{out_name}.json')

        if not os.path.isdir('out/txt'):
            os.makedirs('out/txt')
        with open(f'out/txt/{out_name}.txt', 'w', encoding='utf-8') as f:
            for sent in self.total_sents:
                f.write(f'{sent}\n')
        print(f'문장 파일 저장 완료 --> out/txt/{out_name}.txt')

    def make_custom_dict(self, dict_name="my_dict_01"):
        """
        지정된 이름으로 사용자 정의 사전을 생성하고 업데이트합니다.
        :param dict_name: 생성할 사용자 정의 사전의 이름
        :return None: 이 함수는 내부 상태를 변경하고 사용자 정의 사전을 업데이트하지만 반환 값이 없습니다.
        """
        custom_dict = self.tagger.custom_dict(dict_name)
        word_dict = {key: None for dict_item in self.total_words for key in dict_item.keys()}
        custom_dict.copy_cp_set(word_dict)
        custom_dict.update()

    def make_join_sentence(self, n_sent=1):
        """
        문장을 n문장이 한 문장이 되도록 단위로 합칩니다.
        :param n_sent: 합칠 문장 개수
        :return join_texts: n문장이 합쳐져서 한 문장이 된 리스트
        """
        len_lines = len(self.total_sents)
        num_itters = len_lines//n_sent + 1 if len_lines%n_sent != 0 else len_lines//n_sent
        join_texts = []
        for i in range(num_itters):
            join_texts.append(" ".join(self.total_sents[n_sent*i:n_sent*(i+1)]))
        return join_texts

    def analyze_morpheme(self, out_name, dict_name="my_dict_01", join_n_sent=1):
        """
        사용자 정의 사전을 바탕으로 형태소 분석을 수행하고 그 결과를 파일로 저장합니다.
        :param out_name: 형태소 분석 결과를 저장할 파일 이름
        :param dict_name: 사용할 사용자 정의 사전의 이름
        :return None: 이 함수는 결과를 파일로 저장하므로 반환 값이 없습니다.
        """
        # 이전 사용자 사전 불러오기
        self.custom_dict = self.tagger.custom_dict(dict_name)
        self.custom_dict.load()
        self.tagger.set_domain('my_dict_01')
        results = []
        join_sents = self.make_join_sentence(n_sent=join_n_sent)
        os.makedirs('out/csv', exist_ok=True)
        # CSV 파일을 위한 준비
        with open(f'out/csv/{out_name}.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['문장번호', '문장', '어절번호', '어절', '형태소태그']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            sentence_number = 0
            for sent in join_sents:
                try:
                    # 형태소 분석 호출
                    res = self.tagger.tag(sent, auto_spacing=False, auto_jointing=False) 
                    res = MessageToDict(res.r)
                    start_idx = 0
                    # 500 문장으로 합친 것을 다시 문장 단위로 분할합니다. 어절을 기준으로 index를 계산합니다.
                    for idx in range(0, len(self.total_sents)):
                        _temp_dict = {}
                        end_idx = start_idx+len(self.total_sents[idx].split())
                        _temp_dict[self.total_sents[idx]] = {}
                        tokens = res['sentences'][0]['tokens'][start_idx:end_idx]
                        _temp_dict[self.total_sents[idx]]['tokens'] = tokens
                        start_idx += len(self.total_sents[idx].split())
                        results.append(_temp_dict)
                        
                        # CSV에 쓰기
                        for word_idx, token in enumerate(tokens):
                            word = token['text']['content']
                            morpheme_tags = token['tagged']
                            token_number = f"{sentence_number}-{word_idx + 1}"
                            writer.writerow({'문장번호': sentence_number, '문장': self.total_sents[idx], '어절번호': token_number, '어절': word, '형태소태그': morpheme_tags})
                        sentence_number += 1
                except Exception as e:
                    logger.error(f'An error occurred during sentence analysis, {sent}: {e}')
                    print(f'An error occurred during sentence analysis, {sent}: {e}')
                    continue

        with open(f'out/json/corpus/{out_name}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # 워드 클라우드 생성 및 시각화 함수
    def create_word_cloud(self, sentences, target_word, n=5, output_dir='out/jpg'):
        """
        함수는 주어진 문장들에서 지정된 단어 주변의 문맥을 기반으로 워드 클라우드를 생성하고 저장합니다.
        :param sentences: 워드 클라우드 생성을 위한 문장들
        :param target_word: 워드 클라우드에서 중심이 될 단어
        :param n: 워드 클라우드를 생성할 때 고려할 문맥의 크기 (기본값 5)
        :param output_dir: 워드 클라우드 이미지를 저장할 디렉토리
        :return None: 워드 클라우드는 이미지 파일로 저장되므로 반환 값이 없습니다.
        """
        os.makedirs('out/jpg', exist_ok=True)
        # N-gram 추출
        word_freq = Counter()
        for sentence in sentences:
            tokens = sentence.split()
            if target_word in tokens:
                index = tokens.index(target_word)
                start = max(0, index - (n-1)//2)
                end = min(len(tokens), index + (n-1)//2 + 1)
                window = tokens[start:end]
                # N-gram을 생성하지 않고 단어의 윈도우에서 각 단어를 카운트
                word_freq.update(window)

        # 워드 클라우드로 시각화
        wordcloud = WordCloud(
            font_path='/usr/local/lib/python3.10/dist-packages/matplotlib/mpl-data/fonts/ttf/NanumBarunGothic.ttf',
            width=800,
            height=400,
            background_color='white'
        ).generate_from_frequencies(word_freq)
        output_file = os.path.join(output_dir, f'{target_word}_wordcloud.jpg')
        wordcloud.to_file(output_file)
        print('saved word cloud -> see "out/jpg"')


    def analyze_custom_dict_tokens(self, json_file):
        """
        JSON 파일을 분석하여 사용자 정의 사전에 포함된 단어의 빈도, 다음 토큰의 태그와 빈도, 그리고 클린징된 문장들을 반환합니다.
        :param json_file: 분석할 JSON 파일 경로
        :return custom_word_freq, next_token_tags, all_sentences: 사용자 정의 사전에 포함된 단어의 빈도, 다음 토큰의 태그 및 빈도, 클린징된 문장들
        """
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 중간 처리를 위한 임시 저장소
        custom_word_freq = {}
        next_token_tags = {}
        all_sentences = []  # 모든 문장을 저장할 리스트

        # JSON 데이터 처리
        for d in data:
            for sentence in d.values():
                tokens = sentence['tokens']
                cleaned_sentence = []  # 불용어를 제거한 문장을 위한 리스트
                for i in range(len(tokens)):
                    token = tokens[i]
                    for morpheme in token['morphemes']:
                        token_tag = f"{morpheme['text']['content']}/{morpheme['tag']}"

                        # 정규식을 적용하여 불용어 제거, False면 추가하지 않습니다.
                        if self.check_pattern(token_tag):
                            cleaned_sentence.append(morpheme['text']['content'])
                        # 사용자 사전 단어인지 확인하고, 사용자 사전 단어면 카운트합니다.
                        if 'outOfVocab' in morpheme and morpheme['outOfVocab'] == 'IN_CUSTOM_DICT':
                            word = morpheme['text']['content']
                            custom_word_freq[word] = custom_word_freq.get(word, 0) + 1

                            if i+1 < len(tokens):
                                next_token = tokens[i+1]
                                next_token_tag = next_token['morphemes'][0]['tag']
                                token_tag = f'{next_token["morphemes"][0]["text"]["content"]}/{next_token_tag}'
                                next_token_tags[token_tag] = next_token_tags.get(token_tag, 0) + 1

                # 불용어를 제거한 문장을 all_sentences에 추가
                all_sentences.append(' '.join(cleaned_sentence))

        # 데이터 저장
        os.makedirs('out/csv', exist_ok=True)

        # custom_word_freq를 CSV 파일로 저장
        with open('out/csv/custom_word_freq.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word', 'Frequency'])
            for word, freq in custom_word_freq.items():
                writer.writerow([word, freq])

        # next_token_tags를 CSV 파일로 저장
        with open('out/csv/next_token_tags.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Next Token/Tag', 'Count'])
            for token_tag, count in next_token_tags.items():
                writer.writerow([token_tag, count])

        # 불용어를 제거한 문장을 CSV 파일로 저장
        with open('out/csv/cleaned_sentences.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Cleaned Sentence'])
            for sentence in all_sentences:
                writer.writerow([sentence])

        return custom_word_freq, next_token_tags, all_sentences

    @staticmethod
    def check_pattern(token_tag):
        """
        이 정규 표현식은 세 부분으로 나뉩니다.
        1. /(J|E|S): / 뒤에 J(조사류), E(어미류), S(기호류) 중 하나가 오는 경우를 찾습니다.
        2. \d+/N: 숫자(\d+) 뒤에 /N이 오는 경우를 찾습니다. ex 1/N
        3. ^(하|이|되|있|아니)/V: 문자열의 시작(^)에서 하, 이, 되, 있, 아니 중 하나 뒤에 /V가 오는 경우를 찾습니다.
        :return bool: True, False
        """
        pattern = r'/(J|E|S)|\d+/N|^(하|이|되|있|아니)/V'
        return not re.search(pattern, token_tag)

    def run(self, join_n_sent=1):
        """
        코퍼스 인덱스에 따라서 형태소 분석을 수행합니다.
        """
        for idx, (corpus, k) in enumerate(zip(self.corpus_list, self.corpus_json_dict.keys())):
            print(f'\nStart Analysis index {idx}')
            print(f'target_corpus_directory: {corpus}\nindex: {k}')
            self.extract_word_sent(idx, out_name='은어_데이터')
            self.make_custom_dict(dict_name='my_dict_01')
            self.analyze_morpheme(out_name='형태소분석_결과', dict_name='my_dict_01', join_n_sent=join_n_sent)
            print(f'Finish')
            print(f'---------------------------------------------')
            self.results = [] # 변수 초기화


if __name__ == '__main__':
    corpus_targets = ['data/abbreviation']
    analyzer = CustomDictAnalyzer(corpus_targets)
    analyzer.run(join_n_sent=1)
    custom_word_freq, next_token_tags, cleaned_sentences = analyzer.analyze_custom_dict_tokens('out/json/corpus/형태소분석_결과.json')
    analyzer.create_word_cloud(cleaned_sentences, target_word='고딩', n=3)
    analyzer.create_word_cloud(cleaned_sentences, target_word='직딩', n=3)

