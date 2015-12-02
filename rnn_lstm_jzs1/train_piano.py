##
# -*- coding: utf-8 -*-

# for model
from keras.models import *
from keras.layers.core import *
from keras.layers.recurrent import *

# custom
from data_init import *
from model_util import *

#
import os.path
import random
from datetime import datetime





##
#
# define constant values : dir, file info of training datas
#

print "\n... Set Constant Values ..."

# ================================================================================
#
# 한 번에 모든 midi files 을 이용하여 train 하고 싶을 경우 다음의 경로 사용.
#

# DIR_DATA_SRCs = ["/data/JSB Chorales", "/data/MuseData", "/data/Nottingham", "/data/Piano-midi.de"]
# DIR_TTV = ["/test", "/train", "/valid"]

# examples 1
# JSB chorales 의 train data 로만 training 하고 싶을 경우 - 다음 주석을 풀어줄 것
# path_train = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[0]

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
target_str = ""                         # 폴더의 모든 파일 할거면 파일 이름에 포함된 string 적지 말 것
TARGET_FOLDER = "waltzes"               # training 할 파일이 들어있는 폴더 이름
path_train = "./data_for_train/" + TARGET_FOLDER

# init paths of Waltzes
DIR_WEIGHTS = "./wts_Waltzes/"                      # save weights file
DIR_RESULTS = "./predict_Waltzes/"                  # save debug log
DIR_PREDICTED_MIDI = "./predMidi_Waltzes/"          # save predicted(created) midi file

# file name to save
filename_result_predict = DIR_RESULTS + 'rnn_lstm_predict_{0}.txt'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
# ================================================================================



##
#
# 폴더 없으면 생성하기
#

# train 결과 weights file 저장할 폴더가 없으면 새로 생성
if not os.path.exists(DIR_WEIGHTS):
    print "\n...... Make Folder ...... : ", DIR_WEIGHTS
    os.mkdir(DIR_WEIGHTS)
if not os.path.exists(DIR_RESULTS):
    print "\n...... Make Folder ...... : ", DIR_RESULTS
    os.mkdir(DIR_RESULTS)
    # 폴더 없으면 생성
if not os.path.exists(DIR_PREDICTED_MIDI):
    print "\n...... Make Folder ......", DIR_PREDICTED_MIDI
    os.mkdir(DIR_PREDICTED_MIDI)



##
#
# data set 설정하기 (가져오기)
#

# target_str : 가져오고자 하는 파일에 포함된 string 정보, 위에서 define constant values 구간에서 설정할 것
#              example : target_str = "reels"

print "\n...... Get X, Y, samples_length, seq_length, hidden_size ...... : \n"
print "\tfrom {0},".format(path_train)
print "\ttarget_string : {0}\n".format(target_str)

MidiUtil = Midi_Util()                              # Midi_Util 사용하기

# 경로 내의 모든 파일을 가져오고 싶은 경우
# X, Y, samples_length, seq_length, hidden_size = MidiUtil.get_data_set_from_midi(path_train)

# target_str 을 포함한 파일만 가져오고 싶은 경우
X, Y, samples_length, seq_length, hidden_size = MidiUtil.get_data_set_from_midi(path_train, target_str)



##
#
# model build 하기 : Sequential Model()
#
print "\n...... Start Making Model ......"

#
# model 1 : LSTM
#
model = Sequential()

model.add(LSTM(hidden_size, input_dim=hidden_size, input_length=seq_length, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(hidden_size, input_dim=hidden_size, input_length=seq_length, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(TimeDistributedDense(X.shape[2]))         # hidden size
model.add(Activation('softmax'))                    # softmax layer



# #
# # model 2 : JZS1
# #
# model = Sequential()
#
# model.add(TimeDistributedDense(hidden_size - 1, input_dim=hidden_size, input_length=seq_length))
# model.add(JZS1(hidden_size - 1, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(TimeDistributedDense(hidden_size))
# model.add(Activation('softmax'))



#
# show model
#
ModelUtil.layer_info(model)



##
#
# weigts 값이 있다면 로드해올 것
#

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



##
# set optimizer && loss function
# priority of using optimizer : adam > adadelta > adagrad
print "\n...... Start Compiling Model ......"
model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.compile(loss='categorical_crossentropy', optimizer='adadelta')
# model.compile(loss='categorical_crossentropy', optimizer='adagrad')



##
#
# predict 용 sample 함수들
#
def sample(prob, temperature=1.0):
    # helper function to sample an index from a probability array
    prob = np.log(prob) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    return np.argmax(np.random.multinomial(1, prob, 1))

##
def sample2(prob, temperature=1.0):
    prob = np.log(prob) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    # return np.random.multinomial(random.randint(0,5), prob, 1)
    return np.random.multinomial(random.randint(1,5), prob, 1)

##
# by mansour
def choose_note_index(prob, temperature=1.0, max_chords=4):
    # helper function to sample notes from a probability array
    # 최대 화음은 일단 4개까지로 한정함.
    prob = np.log(prob) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    notes = np.random.choice(prob.shape[0], max_chords, p=prob)
    return np.unique(notes)


def sample_song(seed, midi_file_path):
    print 'Sampling song ...'

    # zero one-hot matrix
    X_predict = np.zeros((1, seq_length, hidden_size))
    X_predict[0, 0, :] = seed  # save seed

    for i in range(seq_length):
        # 예측.
        phat = model.predict(X_predict)
        # note 선택.
        ix = choose_note_index(phat[0, i])
        # end of sequence 가 있으면 종료.
        if np.any(ix == 128):
            break
        # 출력값에 저장.
        if i < seq_length - 1:
            X_predict[0, i + 1, ix] = 1

    # EOS 이전까지로 짤라줌.
    total_beats = (X_predict.sum(axis=-1) > 0).sum()
    X_result = X_predict[0, :total_beats]

    # 미디 파일로 저장.
    MidiUtil.save_midi(midi_file_path, X_result)



##
#
# batch size 및 epoch 값 설정
#
# ================================================================================
training_times = 1000
batch_size_num = seq_length
epoch_num = 100
# ================================================================================



time_start_total = datetime.now()
##
print "\n...... Start Training && Predicting with diversity = 1.0 ......"
with open(filename_result_predict, 'a') as file :
    # for iteration in range(1, 100):
    for iteration in range(1, training_times + 1):
        time_start_iteration = datetime.now()

        print '\n'
        print '-' * 100
        print "Iteration : {0} : is started\n".format(iteration)

        time_start_epochs = datetime.now()

        #
        # train 구간 시작
        #

        # model.fit(X, Y, batch_size=batch_size_num, nb_epoch=epoch_num, show_accuracy=True, shuffle=False)
        model.fit(X, Y, nb_epoch=epoch_num, show_accuracy=True, shuffle=False)

        time_end_epochs = datetime.now()
        print "\n\t{0} Epochs time : {1}".format(epoch_num, time_end_epochs - time_start_epochs)

        # save wts
        filename_wts = DIR_WEIGHTS + "train_piano_wts_seq_model_{0}.wts".format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
        model.save_weights(filename_wts, overwrite=False)
        print "\n\tSaved Weights File : {0}".format(filename_wts)

        #
        # predict 구간 시작
        #

        time_start_predict = datetime.now()
        # iteration 이 10 번 돌 때만 predict 하자
        if iteration % 10 == 0:
            # # for diversity in [0.2, 0.5, 1.0, 1.2]:

            # 트레이닝 세트에서 첫번째 음표 추출
            row = np.random.randint(0, samples_length, 1)
            seed = X[row, X[0].sum(axis=-1) > 0, :][0, :]

            # 추출한 데이터로 샘플송 제작해서 midi 파일로 저장
            FILE_PRED_MIDI = DIR_PREDICTED_MIDI + "rnn_lstm_pred_midi_{0}.mid".format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            sample_song(seed, FILE_PRED_MIDI)

            # debug용
            time_end_predict = datetime.now()
            print "\n\tPredict time : {0}".format(time_end_predict - time_start_predict)

        #
        # predict 구간 끝
        #

        time_end_iteration = datetime.now()
        print "\n\t{0}th Iteration time : {1}".format(iteration, time_end_iteration - time_start_iteration)

        time_passed_total = datetime.now() - time_start_total
        print "\n\tPassed time from starting Training Process : {0}".format(time_passed_total)

        # log 저장
        # save_str =""
        # file.write(save_str)
##
