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
# when you wanna train model with all midi files at once
#

# DIR_DATA_SRCs = ["/data/JSB Chorales", "/data/MuseData", "/data/Nottingham", "/data/Piano-midi.de"]
# DIR_TTV = ["/test", "/train", "/valid"]

# examples 1
# when u want to use "JSB Chorales/train" data for training - uncomment next 2 lines
# path_train = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[0] + DIR_TTV[0]

# examples 2
# when use only bach`s midi files in /data/MuseData/
# target_str = "bach"
# path_train = os.getcwd() + DIR_DATA_SRCs[1] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[1] + DIR_TTV[0]

# examples 3
# when use midi files in /data/Nottingham/
# path_train = os.getcwd() + DIR_DATA_SRCs[2] + DIR_TTV[1]
# path_test = os.getcwd() + DIR_DATA_SRCs[2] + DIR_TTV[0]


# ================================================================================
#
# set directory name what is including data files which you wanna use at triaing
# set string(target_str) what is included in data files which you wanna use at triaing
#
target_str = ""                         # if u use all files at certain directory, don`t set any letters
TARGET_FOLDER = "waltzes"               # directory name which is containing data files for training
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
# If there are not existing directories, make directories using command "mkdir"
#

# directory to save weights file while training is done
if not os.path.exists(DIR_WEIGHTS):
    print "\n...... Make Folder ...... : ", DIR_WEIGHTS
    os.mkdir(DIR_WEIGHTS)
if not os.path.exists(DIR_RESULTS):
    print "\n...... Make Folder ...... : ", DIR_RESULTS
    os.mkdir(DIR_RESULTS)
if not os.path.exists(DIR_PREDICTED_MIDI):
    print "\n...... Make Folder ......", DIR_PREDICTED_MIDI
    os.mkdir(DIR_PREDICTED_MIDI)



##
#
# set Data Set (load Data Set)
#
# target_str : string info what is included at files u want to load for training
#              set this variable at block "define constant values"
#              example : target_str = "reels"

print "\n...... Get X, Y, samples_length, seq_length, hidden_size ...... : \n"
print "\tfrom {0},".format(path_train)
print "\ttarget_string : {0}\n".format(target_str)

MidiUtil = Midi_Util()                              # use Midi_Util

# if u want to load all files at path_train
# X, Y, samples_length, seq_length, hidden_size = MidiUtil.get_data_set_from_midi(path_train)

# if u want to load filese which is including target_str
X, Y, samples_length, seq_length, hidden_size = MidiUtil.get_data_set_from_midi(path_train, target_str)



##
#
# Build Model : Sequential Model()
#
print "\n...... Start Making Model ......"

#
# Model 1 : LSTM (Long Short Term Memory)
#
model = Sequential()

model.add(LSTM(hidden_size, input_dim=hidden_size, input_length=seq_length, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(hidden_size, input_dim=hidden_size, input_length=seq_length, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(TimeDistributedDense(X.shape[2]))         # hidden size
model.add(Activation('softmax'))                    # softmax layer



# #
# # Model 2 : JZS1
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
# if there is weigts, load weights
#

# load automatically : load weights which is containing the latest weights infomation
try:
    wts_list = os.listdir(DIR_WEIGHTS)
    if len(wts_list) != 0:
        wts_list.sort()
        model.load_weights(DIR_WEIGHTS + wts_list[-1])
        print "\n...... Loaded weights file : {0} ......".format(wts_list[-1])
except:
    pass

# # load passively
# #
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
# functions for sampling predict data
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
    # maximum chords are limited 4 notes
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
        # predict
        phat = model.predict(X_predict)
        # choice note
        ix = choose_note_index(phat[0, i])
        # break if end of sequence (EOS) is found
        if np.any(ix == 128):
            break
        # save at predict variable
        if i < seq_length - 1:
            X_predict[0, i + 1, ix] = 1

    # cut out EOS
    total_beats = (X_predict.sum(axis=-1) > 0).sum()
    X_result = X_predict[0, :total_beats]

    # save predicted data with midi format
    MidiUtil.save_midi(midi_file_path, X_result)



##
#
# set batch size, epoch values
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
        # train : start
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
        # predict : start
        #

        time_start_predict = datetime.now()
        # do prediction at every 10th iteration
        if iteration % 10 == 0:
            # # for diversity in [0.2, 0.5, 1.0, 1.2]:

            # get first note at training set
            row = np.random.randint(0, samples_length, 1)
            seed = X[row, X[0].sum(axis=-1) > 0, :][0, :]

            # with above note, make the sample song and save this song with midi format
            FILE_PRED_MIDI = DIR_PREDICTED_MIDI + "rnn_lstm_pred_midi_{0}.mid".format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            sample_song(seed, FILE_PRED_MIDI)

            # debug
            time_end_predict = datetime.now()
            print "\n\tPredict time : {0}".format(time_end_predict - time_start_predict)

        #
        # predict : end
        #

        time_end_iteration = datetime.now()
        print "\n\t{0}th Iteration time : {1}".format(iteration, time_end_iteration - time_start_iteration)

        time_passed_total = datetime.now() - time_start_total
        print "\n\tPassed time from starting Training Process : {0}".format(time_passed_total)

        # save log
        # save_str =""
        # file.write(save_str)
##
