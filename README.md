# Text-Analysis
* 이 프로젝트는 바른 형태소 분석기를 이용하여 형태소 분석의 기초에 대한 실습을 제공합니다.
* 사용자사전, 토크나이징 활용

## 사전 준비사항
* 가상환경 설정
  * 현재 프로젝트에 파이썬 가상환경 생성
  * 가상환경 활성화
  * pip upgrade
  * 라이브러리 설치

```
python3 -m venv venv 
source venv/bin/activate 
pip install --upgrade pip 
pip install -r requirements.txt 
```
* 파이썬 인터프리터(단축키: 컨트롤+쉬프트+P) 선택 -> venv 선택

## 사용자정의 사전 적용하기

* CustomDictAnalyzer 클래스 설명
  * CustomDictAnalyzer 클래스는 바른 형태소 분석기를 사용하여 텍스트 분석을 수행합니다. 이 클래스는 여러 코퍼스 파일을 분석하여 형태소 분석, 사용자 정의 사전 생성, 워드 클라우드 생성 등의 기능을 제공합니다.

## Word2Vec 임베딩 시각화

Word2Vec 모델의 단어 임베딩을 시각화하는 과정을 담고 있습니다. 주요 단계는 다음과 같습니다:

1. **한글 폰트 설정**: 한글 텍스트가 올바르게 표시되도록 폰트를 설정합니다.
2. **t-SNE 변환**: Word2Vec 모델의 임베딩을 2차원으로 변환합니다.
3. **가장 가까운 이웃 찾기**: 특정 단어들에 대한 가장 가까운 이웃 단어들을 찾습니다.
4. **시각화**: 임베딩과 이웃 단어들을 시각화합니다.

### tensorboard 실행

```
tensorboard --logdir logs
```