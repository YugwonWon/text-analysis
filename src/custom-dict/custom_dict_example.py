import json
import os
import glob
import csv
from bareunpy import Tagger
from google.protobuf.json_format import MessageToDict

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
        self.corpus_wav_dict = {}
        for idx, corpus in enumerate(corpus_list):
            wavs = glob.glob(os.path.join(corpus, '**/*.wav'), recursive=True) # file 이름 규칙은 변경될 수 있다.
            json_files = glob.glob(os.path.join(corpus, '**/*.json'), recursive=True)
            self.corpus_json_dict[f'{str(idx)}_json'] = json_files # 요청된 corpus 리스트 순서대로 번호를 부여한다.
            self.corpus_wav_dict[f'{str(idx)}_wavs'] = self.mapping_name(wavs)

        self.results = []

    def mapping_name(self, target_dir_files):
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
        custom_dict = self.tagger.custom_dict(dict_name)
        word_dict = {key: None for dict_item in self.total_words for key in dict_item.keys()}
        custom_dict.copy_cp_set(word_dict)
        custom_dict.update()
    
    def analyze_morpheme(self, out_name, dict_name="my_dict_01"):
        """
        형태소 분석을 수행합니다.
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
    
    def analyze_custom_dict_tokens(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        custom_word_freq = {}
        next_token_tags = {}
        for d in data:
            for sentence in d.values():
                tokens = sentence['tokens']
                for i in range(len(tokens)):
                    token = tokens[i]
                    for morpheme in token['morphemes']:
                        if 'outOfVocab'in morpheme and morpheme['outOfVocab'] == 'IN_CUSTOM_DICT':
                            word = morpheme['text']['content']
                            custom_word_freq[word] = custom_word_freq.get(word, 0) + 1

                            if i+1 < len(tokens):
                                next_token = tokens[i+1]
                                next_token_tag = next_token['morphemes'][0]['tag']
                                token_tag = f'{next_token["morphemes"][0]["text"]["content"]}/{next_token_tag}'
                                next_token_tags[token_tag] = next_token_tags.get(token_tag, 0) + 1
        
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
                
        return custom_word_freq, next_token_tags
        
    
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
    # analyzer.run()
    custom_word_freq, next_token_tags = analyzer.analyze_custom_dict_tokens('out/json/corpus/형태소분석_결과.json')
    print(custom_word_freq)
    print(next_token_tags)
    