# -*- coding: utf-8 -*-

from keras.callbacks import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.models import *

from midi_util import *
from model_util import *

np.random.seed(275129)

#
# 파라미터 설정.
#

input_dim = 129
hidden_size = 128
seq_length = 350

#
# 데이터 만들기.
#

print 'Loading data ...'

samples = 5
X = np.zeros((samples, seq_length + 1, input_dim), dtype='int32')

# 미디 파일을 읽어서 one-hot matrix 로 바꾸어줌.
for i in range(samples):
    om = MidiUtil.load_midi('canon14.mid')
    # left zero-padding
    start = max(0, seq_length - om.shape[0] + 1)
    end = min(om.shape[0], seq_length + 1)
    X[0, start:, :] = om[:end, :]

# X_train 과 Y_train 으로 분리
# (Y_train 이 한칸 더 앞서게 배치함.)
X_train = X[:, :-1, :]
Y_train = X[:, 1:, :]

#
# 모델링.
#

model = Sequential()

model.add(TimeDistributedDense(hidden_size, input_dim=input_dim, input_length=seq_length))
model.add(JZS1(hidden_size, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(input_dim))
model.add(Activation('softmax'))

ModelUtil.layer_info(model)

print 'Building model ...'
model.compile(loss='categorical_crossentropy', optimizer='adam')
print 'Finished model building ...'

# 저장된 모델 로딩.
try:
    model.load_weights('music_rnn.hdf5')
except:
    pass


#
# 유틸리티 함수 정의.
#

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
    X_predict = np.zeros((1, seq_length, input_dim))
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


#
# 트레이닝
#

# 모델 저장용 콜백
cp = ModelCheckpoint('music_rnn.hdf5')

total_iter = 1000
for iter in range(total_iter):
    # 트레이닝
    print 'Iteration %d / Total %d' % (iter, total_iter)
    model.fit(X_train, Y_train, nb_epoch=100, callbacks=[cp])

    #
    # 중간 결과로 작곡해 보기.
    #

    # 트레이닝 세트에서 첫번째 음표 추출.
    row = np.random.randint(0, samples, 1)
    seed = X_train[row, X_train[0].sum(axis=-1) > 0, :][0, :]

    # 샘플송 제작해서 midi 파일로 저장.
    sample_song(seed, 'sample%03d.mid' % iter)
