music-rnn
========
Machine Learning / Deep Learning - Compose Waltz
>
- midi files 를 data set 으로 사용하여 midi files 의 notes(음정보) 에 대하여 training 한 후
- train 된 스타일로 새로운 곡을 컴퓨터가 만들어내도록하는
- Deep Learning 을 이용한 간단한 Python Code

Machine Learning / Deep Learning 을 이용한 간단한 음악 작곡
--------
>
- Deep Learning 에 대해서 잘 모른다해도
- 위의 코드들과 아래의 Dependencies 를 참조하여 수정 및 테스트를 반복하다보면
- 수식에 대한 지식여부와는 관계없이
- 기본적으로 간단한 Deep Learning 을 위한 Python Code 를 **직접** 돌려볼 수 있을 것임.



Dependencies
--------
### OpenBLAS
- an optimized BLAS(Basic Linear Algebra subroutines) library based on [GotoBLAS2](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2) 1.13 BSD version.
- NumPy 사용을 위해서 필요
- OpenBLAS 를 install 하기 위해서는 gfortran 을 먼저 설치할 것
- http://www.openblas.net/
- https://github.com/xianyi/OpenBLAS/wiki

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
- Zulko 라는 사람이 Piano Rolls 를 Python, LilyPond 를 이용하여 Piano Roll Movie 와 midi 파일의 음악을 악보로 옮기기 위해 만든 프로젝트
- http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/
- http://zulko.github.io/unroll/
- https://github.com/Zulko/unroll

### music21
- a toolkit for computer-aided musicology (MIT)
- http://web.mit.edu/music21/
- http://web.mit.edu/music21/doc/index.html



Train / Test 에 사용한 Dataset
--------
### Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription
- http://www-etud.iro.umontreal.ca/~boulanni/icml2012
- 기본적으로 위의 midi files data set 을 사용하여 train 및 test 를 수행했음.
- 위의 data set 을 이용하여 직접  training 을 해보고 싶을 경우 아래의 **Usage : Data Set 바꾸기** 참고


Usage
--------
### Data Set, Source Code 구성
```
music-rnn
├── README.md
└── rnn_lstm_jzs1
    ├── data
    │   ├── JSB_Chorales
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   ├── MuseData
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   ├── Nottingham
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   └── Piano-midi.de
    │       ├── test
    │       ├── train
    │       └── valid
    ├── data_for_train
    │   └── waltzes
    │       ├── waltzes_simple_chords_1.mid
    │       ├── waltzes_simple_chords_2.mid
    │       ├── waltzes_simple_chords_3.mid
    │       ├── waltzes_simple_chords_6.mid
    │       └── waltzes_simple_chords_7.mid
    ├── predict_Waltzes
    ├── predMidi_Waltzes
    │   └── bach_lstm_pred_midi_2015.11.11.15:32:34.mid
    ├── wts_Waltzes
    │   └── train_piano_wts_seq_model_2015.11.11.23:52:29.wts
    ├── __init__.py
    ├── data_init.py
    ├── model_util.py
    ├── train_piano.py
    └── readme.md 

```


### 기본 사용 방법
- 기본적으로 train_piano.py 파일을 python 으로 바로 실행시킬 경우
  * /music-rnn/rnn_lstm_jzs1/wts_Waltzes/train_piano_wts_seq_model_2015.11.11.23:52:29.wt
  * 위의 weights 파일을 LSTM(Long Short Term Memory) training model 이 load
  * 그 후 music-rnn/rnn_lstm_jzs1/data_for_train/waltzes 폴더 안에 있는 midi 형식으로 된 5개의 waltz 곡을 이용하여 training 시작

```python
$ git clone https://github.com/jamonglab/music-rnn.git
$ cd music-rnn
$ python train_piano.py
```
- git clone(복사) 명령어를 이용해서 github 에서 music-rnn project 를 download
  * git/github 에 대한 설명 및 사용법
    * [완전 초보를 위한 깃허브](https://nolboo.github.io/blog/2013/10/06/github-for-beginner/)
    * [git - 간편 안내서](https://rogerdudler.github.io/git-guide/index.ko.html)
    * [git-scm.com](http://git-scm.com/book/ko/v2/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-%EB%B2%84%EC%A0%84-%EA%B4%80%EB%A6%AC%EB%9E%80%3F)
- clone 이 저장된 music-rnn 폴더로 이동
- train_piano.py 를 python 을 이용하여 컴파일 및 실행
  
  * 작곡된 waltz 곡 저장되는 폴더
    * train_piano.py 소스 코드를 보면
    ```
    if iteration % 10 == 0:
    ```
    * training 실행 시 iteration 을 10번 반복할 때 마다 machine 이 배운 결과 생성된 weight 를 이용하여 새로운 waltz 곡을 composing 하여
    ```
    DIR_PREDICTED_MIDI = "./predMidi_Waltzes/"          # save predicted(created) midi file
    ```
    * /predMidi_Waltzes 폴더에 저장하게 된다.

### weights 파일을 직접 구해보기 / 다른 weights file 이용하기
- 기본적으로 train_piano.py 소스 코드는 다음과 같이 weights 파일이 존재한다면 가장 최근의 파일을 load 하여 model 에 적용되도록 되어있다.
```python
# 자동 : 가장 최근의 마지막 wts 가져오기
try:
    wts_list = os.listdir(DIR_WEIGHTS)
    if len(wts_list) != 0:
        wts_list.sort()
        model.load_weights(DIR_WEIGHTS + wts_list[-1])
        print "\n...... Loaded weights file : {0} ......".format(wts_list[-1])
except:
    pass

# # 수동 : 가져올 wts 파일을 파일이름으로 지정해주기
# # waltzes wts file ==> loss: 1.5240 - acc: 0.3573
# # ./wts_Waltzes/train_piano_wts_seq_model_2015.11.11.23:52:29.wts
# filename_wts = "train_piano_wts_seq_model_2015.11.11.23:52:29.wts"
# try:
#     model.load_weights(DIR_WEIGHTS + filename_wts)
#     print "\n... Loaded weights file : {0} ...".format(filename_wts)
# except:
#     pass
```
- 직접 weights 파일을 구해보고 싶은 경우
  * music-rnn/rnn_lstm_jzs1/wts_Waltzes 폴더의 train_piano_wts_seq_model_2015.11.11.23:52:29.wts 파일을 삭제하거나 임의의 폴더로 옮겨놓은 후
  * 위의 **기본 사용 방법**을 그대로 따라서 실행하면 된다.
- 다른 weights 파일을 이용하고 싶은 경우
  * 위의 source code 중 자동이라고 comment 된 블럭을 comment 처리 한 후, 수동 comment 의 블럭의 comment 를 해제하고
  * 이용하고자 하는 weights 파일을 DIR_WEIGHTS 폴더에 위치시킨 후,
  * filename_wts = "파일 이름" 을 적어주고,
  * 위의 **기본 사용 방법**을 그대로 따라서 실행하면 된다.

### Data Set 바꾸기
- Train / Test 에 사용했던 Dataset 인 Modeling Temporal Dependencies in High-Dimensional Sequences 의 data set 을 사용해보고 싶은 경우
- http://www-etud.iro.umontreal.ca/~boulanni/icml2012
- 위의 링크로 이동하여 Source 파일들을 다운받은 후,
  - Piano-midi.de1 : Source (124 files, 951 KB) or Piano-roll (7.1 MB)
  - Nottingham2 : Source (1037 files, 676.1 KB) or Piano-roll (23.2 MB)
  - MuseData3 : Source (783 files, 3.0 MB) or Piano-roll (30.1 MB)
  - JSB Chorales : Source (382 files, 210 KB) or Piano-roll (2.0 MB)
- 압축을 풀고 music-rnn/rnn_lstm_jzs1/data 폴더 아래에 다음과 같이 위치시켜준 후,
```
music-rnn
└── rnn_lstm_jzs1
    ├── data
    │   ├── JSB_Chorales
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   ├── MuseData
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   ├── Nottingham
    │   │   ├── test
    │   │   ├── train
    │   │   └── valid
    │   └── Piano-midi.de
    │       ├── test
    │       ├── train
    │       └── valid
```
- 예를 들어 Data Set 중 JSB_Chorales set 을 이용하고 싶다면 train_piano.py 의 source code 중 data set 을 설정하는 부분의 code 를 다음과 같이 수정해준다.
```python
# ================================================================================
#
# 한 번에 모든 midi files 을 이용하여 train 하고 싶을 경우 다음의 경로 사용.
#

DIR_DATA_SRCs = ["/data/JSB_Chorales", "/data/MuseData", "/data/Nottingham", "/data/Piano-midi.de"]
DIR_TTV = ["/test", "/train", "/valid"]

# examples 1
# JSB_chorales 의 train data 로만 training 하고 싶을 경우 - 다음 주석을 풀어줄 것
path_train = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[1]
path_test = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[0]

# examples 2
# /data/MuseData/ 폴더의 bach midi files 로만 train
# target_str = "bach"
# path_train = os.getcwd() + DIR_DATA_SRCs[1] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[1] + DIR_TTV[0]

# examples 3
# /data/Nottingham/ 폴더 midi files 로만 train
# path_train = os.getcwd() + DIR_DATA_SRCs[2] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[2] + DIR_TTV[0]


# ================================================================================
#
# 가져올 sample 이 포함된 폴더 이름과 가져올 파일에 포함된 string setting
#
# target_str = ""                         # 폴더의 모든 파일 할거면 파일 이름에 포함된 string 적지 말 것
# TARGET_FOLDER = "waltzes"               # training 할 파일이 들어있는 폴더 이름
# path_train = "./data_for_train/" + TARGET_FOLDER

# init paths of Waltzes
# DIR_WEIGHTS = "./wts_Waltzes/"                      # save weights file
# DIR_RESULTS = "./predict_Waltzes/"                  # save debug log
# DIR_PREDICTED_MIDI = "./predMidi_Waltzes/"          # save predicted(created) midi file
DIR_WEIGHTS = "./wts_JSB/"                      # save weights file
DIR_RESULTS = "./predict_JSB/"                  # save debug log
DIR_PREDICTED_MIDI = "./predMidi_JSB/"          # save predicted(created) midi file

# file name to save
filename_result_predict = DIR_RESULTS + 'rnn_lstm_predict_{0}.txt'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
# ================================================================================

```
- 추가로 어느 한 폴더의 공통 string을 포함하는 파일만 training 에 이용하고 싶을 경우 위 소스 코드 중 ```target_str = ""``` 부분에 원하는 파일들이 공통으로 가지고 있는 string 을 적어주면 된다.
  * 예를 들어, 폴더에 'bach001.mid', 'bach002.mid', ... 과 같이 **"bach" ** 를 공통으로 파일 이름에 사용하고 있고,
  * 이 bach 의 음악들을 trainging 에 사용하고 싶다면,
  * ```target_str = "bach"```
  * 위와 같이 소스 코드를 수정해주면 된다.
