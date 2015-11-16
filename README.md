# music-rnn
Machine Learning / Deep Learning - Compose Waltes


Machine Learning / Deep Learning 을 이용해서 간단하게 음악 작곡하기
======================================================================
>
- midi files 를 data set 으로 사용하여 midi files 의 notes(음정보) 에 대하여 training 한 후
- train 된 스타일로 새로운 곡을 컴퓨터가 만들어내도록하는
- Deep Learning 을 이용한 간단한 Python Code



- Deep Learning 에 대해서 잘 모른다해도
- 위의 코드들과 아래의 Dependencies 를 참조하여 수정 및 테스트를 반복하다보면
- 수식에 대한 지식여부와는 관계없이
- 기본적으로 간단한 Deep Learning 을 위한 Python Code 를 **직접** 돌려볼 수 있을 것임.



Dependencies
--------
### NumPy
- 수치적 배열과 고급 데이터 분석을 편리하게 해주는 Python Package
- SciPy 와 함께 널리 쓰임.
- http://www.numpy.org/

### SciPy
- http://www.scipy.org/

### Theano
- GPU 를 활용한 연산/분석 코드를 작성할 수 있도록 도와주고, 
- 다차원 배열을 사용한 수학 식을 정의, 최적화, 풀이하는 Python Library
- http://deeplearning.net/software/theano/#

### Keras
- Python 으로 작성된 Neural Network Library
- https://github.com/fchollet/keras
- http://keras.io/

### UnRoll
- Zulko 라는 사람이 Piano Rolls 를 Python, LilyPond 를 이용하여 Movie 와 midi 파일의 음악을 악보로 옮기기 위해 만든 프로젝트
- http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/
- http://zulko.github.io/unroll/
- https://github.com/Zulko/unroll

### music21
- a toolkit for computer-aided musicology (MIT)
- http://web.mit.edu/music21/
- http://web.mit.edu/music21/doc/index.html



Train / Test 에 사용했던 Dataset
--------
### Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription
- http://www-etud.iro.umontreal.ca/~boulanni/icml2012
- 기본적으로 위의 midi files data set 을 사용하여 train 및 test 를 수행했음.
- 위의 자료를 사용하고자 하는 경우 소스를 다운 받은 후, 압축을 풀고 juce/data 아래에 data set 을 위치시킬 것.
- Waltzes 작곡을 위한 overfitted 된 weight 는 juce/wts_Waltzes 폴더에 포함되어있음.
- 직접 overfitted 된 weight 를 구하고 싶은 경우
  * juce/data_for_train/ 에 포함되어 있는 waltes midi files 를 이용할 것.



Usage
--------
```sh
$ python train_piano.py
```
- 세부적인 조절은 소스 코드의 주석을 참조할 것
