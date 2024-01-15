import json
import os
import glob
import csv
import re
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
        self.tagger = Tagger(self.config['bareun-api-key'], 'localhost')
        
        # json파일 및 wav파일 목록을 만듭니다.
        self.corpus_list = corpus_list
        self.corpus_json_dict = {}
        # self.corpus_wav_dict = {}
        for idx, corpus in enumerate(corpus_list):
            # wavs = glob.glob(os.path.join(corpus, '**/*.wav'), recursive=True) # file 이름 규칙은 변경될 수 있다.
            json_files = glob.glob(os.path.join(corpus, '**/*.json'), recursive=True)
            self.corpus_json_dict[f'{str(idx)}_json'] = json_files # 요청된 corpus 리스트 순서대로 번호를 부여한다.
            # self.corpus_wav_dict[f'{str(idx)}_wavs'] = self.mapping_name(wavs)

        self.results = []

    def mapping_name(self, target_dir_files):
        """
        파일 경로의 목록을 받아 파일 이름을 키로, 전체 파일 경로를 값으로 하는 사전을 생성합니다.
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
    
    def analyze_morpheme(self, out_name, dict_name="my_dict_01"):
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
        for sent in self.total_sents:
            try:
                _temp_dict = {}
                res = self.tagger.tag(sent)
                res = MessageToDict(res.r)
                _temp_dict[res['sentences'][0]['text']['content']] = {}
                _temp_dict[res['sentences'][0]['text']['content']]['tokens'] = res['sentences'][0]['tokens']
                results.append(_temp_dict)
            except:
                print(f'analyze error! sent: {sent}')
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
                        if self.check_pattern(token_tag):  # 불용어 제거
                            cleaned_sentence.append(morpheme['text']['content'])

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
        pattern = r'/(J|E|S|E)|\d+/N|^(하|이|되|있|아니)/V'
        return not re.search(pattern, token_tag)    
    
    def run(self):
        """
        코퍼스 인덱스에 따라서 형태소 분석을 수행합니다.
        """
        for idx, (corpus, k) in enumerate(zip(self.corpus_list, self.corpus_json_dict.keys())):
            print(f'\nStart Analysis index {idx}')
            print(f'target_corpus_directory: {corpus}\nindex: {k}')
            self.extract_word_sent(idx, out_name='은어_데이터')
            self.make_custom_dict(dict_name='my_dict_01')
            self.analyze_morpheme(out_name='형태소분석_결과', dict_name='my_dict_01')
            print(f'Finish')
            print(f'---------------------------------------------')
            self.results = [] # 변수 초기화
        

if __name__ == '__main__':
    corpus_targets = ['data/abbreviation']
    analyzer = CustomDictAnalyzer(corpus_targets)
    analyzer.run()
    custom_word_freq, next_token_tags, cleaned_sentences = analyzer.analyze_custom_dict_tokens('out/json/corpus/형태소분석_결과.json')
    analyzer.create_word_cloud(cleaned_sentences, target_word='고딩', n=3)
    analyzer.create_word_cloud(cleaned_sentences, target_word='직딩', n=3)

    