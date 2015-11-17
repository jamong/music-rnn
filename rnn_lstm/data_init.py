##
# -*- coding: utf-8 -*-
"""
    Unroll :
        Python package for Piano roll transcription to sheet music
            https://github.com/Zulko/unroll
            http://zulko.github.io/unroll/
    Music21 :
        What is music21?
            http://web.mit.edu/music21/
        Music21 Documentation
            http://web.mit.edu/music21/doc/index.html
"""



# for dataSet
from unroll.KeyStrikes import KeyStrikes
import music21
import numpy as np
import os.path



class Midi_Util(object):
    def __init__(self):
        # index 나누는 기준
        # midi file 을 변환하고 나면 일단 1024(4분음표) 기준으로 바뀐다.
        # tick_step = mf.ticksPerQuarterNote / 2            # tick_setp 계산 공식
        # tick_step 을 512 로 init 했으나, 실제로는 midi 파일을 읽는 부분에서 자동으로 다시 계산함.
        self.tick_step = 512

        self.samples_num = 0                                # samples number, will initialized
        self.seq_length = 0                                 # sequence length, will initialized
        # hidden size
        self.note_dim = 128                                 # midi 파일 규격 : 총 128 음표
        self.data_dim = self.note_dim + 1                   # EOS 추가
        self.hidden_size = self.data_dim                    # just naming, data dimention == hidden size


    def get_data_set_from_midi(self, path_data, target_str=""):
        """
        .
        :param path_data: midi 파일들이 저장된 directory path 정보
        :param target_str: directory path 에서 불러올 target 파일들이 공통으로 가지고 있는 string
        :return: X train data set, Y train data set, samples number, sequence length, hidden size
        """
        # target samples list 가져오기
        self.samples_num, samples_list = self.get_samples_list(path_data, target_str)

        # data file 중에 가장 긴 파일의 길이를 가져와서 seq_length 지정하기
        self.seq_length = self.get_data_set_max_seq_length(path_data, samples_list)

        # for debug
        print "\tdebug --------> self.seq_length : {0}\n\n".format(self.seq_length)

        # data_set 초기화
        X_train = np.zeros((self.samples_num, self.seq_length, self.hidden_size), dtype=int)
        Y_train = np.copy(X_train)

        #
        # 모든 파일의 데이터를 파일별로 convert 해서 one-hot vector 로 저장하기
        #

        samples_cnt = 0                                         # sample 시작 번호 세팅 : 몇 번째 midi 파일인가.
        for filename in samples_list:
            one_X_train = self.load_midi(path_data, filename)   # midi file 하나의 notes 정보 가져오기

            # 추출한 midi file 하나의 note 정보를 가진 matrix 를 train data set 인 X_train 에 집어 넣기.
            # left zero-padding
            X_train[samples_cnt][self.seq_length - one_X_train.shape[0] + 1:][:] = np.copy(one_X_train[:-1][:])

            samples_cnt += 1

        # Y_train 세팅, X_train 보다 한 칸 앞서게
        Y_train[:][:-1][:] = np.copy(X_train[:][1:][:])

        # for debug
        # print "\n\tX_train.shape : {0},        Y_train.shape : {1}".format(X_train.shape, Y_train.shape)
        # print "\tX_train : ", X_train
        # print "\tY_train : ", Y_train
        # print "\tsamples_num : ", self.samples_num
        # print "\tseq_length : ", self.seq_length
        # print "\tdata_dim : ", self.data_dim

        return X_train, Y_train, self.samples_num, self.seq_length, self.data_dim


    def get_data_set_max_seq_length(self, path_data, samples_list):
        """
        load 할 data 들 중 가장 긴 play time 을 가진 midi 파일을 찾아 기준이 될 index 길이를 반환
        :param path_data:
        :param samples_list:
        :return: seq_length 의 기준이 될 midi 파일의 notes index 번호
        """
        # for debug
        whatislongestfile = ""
        print "\n...... get info : longest sequence length ......"
        max_ks_time = 0
        mfticksperquarternote = 0
        for filename in samples_list:
            ks, mfticksperquarternote = self.load_midi(path_data, filename, get_longest_length=True)

            # 가장 time 이 긴 파일의 seq_length 구하기
            now_ks_time = ks.keystrikes[-1]['time'] + ks.keystrikes[-1]['duration'] * mfticksperquarternote
            if max_ks_time < now_ks_time:
                max_ks_time = now_ks_time
                whatislongestfile = filename

        tick_step = mfticksperquarternote / 2              # 틱 간격은 8분의 1음표 단위로
        longest_seq_length = max_ks_time / tick_step + 1  # EOS 를 위해서 index 1 추가

        print "\n\tLongest File name : {0}, sequence length : {1}\n\n".format(whatislongestfile, longest_seq_length)
        return longest_seq_length



    def get_samples_list(self, path_data, target_str=""):
        """
        midi files 가 저장된 path 와 target file 의 string 을 받아서
        data 로 사용할 midi files 의 갯수와 midi files`s name 을 return
        :param path_data: midi 파일들이 들어있는 폴더의 경로
        :param target_str: path_data 에 들어있는 midi 파일들 중 이름을 기준으로 가져올 파일을 선택하기
        :return: 사용할 midi files 의 갯수, file name list 를 return
        """
        print "\n...... get samples list ......"
        # path 에 있는 파일 목록
        samples_list = os.listdir(path_data)

        # target_str 에 맞는 파일 이름만 걸러내기
        if target_str != "":
            temp_list_train_files = []
            for item in samples_list:
                if target_str in item:
                    temp_list_train_files.append(item)
            samples_list = temp_list_train_files

        # 파일이름순 정렬
        samples_list.sort()
        # sample 갯수 저장
        self.samples_num = len(samples_list)

        print "\n\tReturn {0} file name list.".format(len(samples_list))
        return len(samples_list), samples_list


    def load_midi(self, path_data, filename, get_longest_length=False):
        """
        .
        :param path_data: midi files 가 저장되어있는 path
        :param filename: midi file name
        :param get_longest_length: longest play time 을 가진 파일의 길이만 받을지 말지.
        :return:
        """
        # read midi file
        mf = music21.midi.MidiFile()
        path_midifile = path_data + '/' + filename
        mf.open(path_midifile)
        mf.read()
        mf.close()

        # midi -> music21 stream -> midi
        # mf 파일의 ticksperQuarterNote 를 1024 기준으로 바꿔주기
        s = music21.midi.translate.midiFileToStream(mf)
        mf = music21.midi.translate.streamToMidiFile(s)

        #
        # 트랙별 루프 돌면서 note 정보 읽기
        #
        #   트랙별 event 의 구조는 다음과 같다.
        #   <MidiTrack 1 -- 1092 events
        #   <MidiEvent DeltaTime, t=0, track=1, channel=1>
        #   .....
        #   <MidiEvent TIME_SIGNATURE, t=0, track=1, channel=1, data='\x04\x02\x18\x08'>
        #   <MidiEvent DeltaTime, t=0, track=1, channel=1>
        #   <MidiEvent NOTE_ON, t=0, track=1, channel=1, pitch=76, velocity=105>
        #   <MidiEvent DeltaTime, t=512, track=1, channel=1>
        #   <MidiEvent NOTE_OFF, t=0, track=1, channel=1, pitch=76, velocity=0>
        #
        result = []
        for i in range(len(mf.tracks)):
            t = 0
            for e in mf.tracks[i].events:
                if e.isDeltaTime() and e.time is not None:
                    t += e.time
                elif e.isNoteOn() and e.pitch is not None and e.velocity != 0:
                    result.append({'time': t, 'note': e.pitch})

        # unroll.KeyStrikes 이용
        ks = KeyStrikes(result)
        # duration 가지도록 list 변환
        ks = ks.quantized(mf.ticksPerQuarterNote)

        # 가장 sequence 가 긴 파일의 sequence 를 알고 싶을 때
        # get_longest_length = True 로 setting 해주면 가장 긴 sequence 와
        # mf.ticksPerQuarterNote 만 반환.
        if get_longest_length:
            return ks, mf.ticksPerQuarterNote

        #
        # one-hot vector 형식으로 변환
        #

        # tick 간격 = 1/8분 음표 단위로
        self.tick_step = mf.ticksPerQuarterNote / 2
        # 총 틱 카운트
        total_tick = ks.keystrikes[-1]['time'] + ks.keystrikes[-1]['duration'] * mf.ticksPerQuarterNote
        # 총 sequence 갯수
        total_beat = total_tick / self.tick_step

        # 하나의 midi 파일의 one hot vector 를 저장할 임시 matrix
        one_X_train = np.zeros((total_beat + 1, self.data_dim), dtype='int32')

        # 1024 기준으로 convert 된 건반 정보를 루프를 돌면서 디코딩
        for strike in ks.keystrikes:
            # 1/8 음표 단위의 index
            idx = strike['time'] / self.tick_step
            # 1/8 음표 단위 기준 note 의 지속시간 (index 를 몇개 채울거니? 를 계산)
            dur = int(strike['duration'] * 2)

            # note 별로 one-hot vector 채우기
            for note in strike['note']:
                # 지속시간 만큼 making
                for i in range(dur):
                    one_X_train[idx + i, note] = 1

        # 끝에 EOS 추가 = EOS 는 data_dim[-1] = 1 인 경우.
        one_X_train[-1, self.note_dim] = 1

        print '\tMidi file(=%s) successfully loaded. Total 1/8 beats = %d' % (path_midifile, total_beat)

        return one_X_train


    # 이 training 에서는 사용하지 않음. 예비용.
    def get_ix2note(self):
        print "\n...... get matrix of index to note number ......"
        ix2note = []
        # max_note_num + 1(108) + 1(EOS)
        # for note in range(self.min_note_num, self.max_note_num + 1 + 1, 1):
        for note in range(self.data_dim):
            ix2note.append(note + 1)

        return ix2note


    def save_midi(self, midi_file_path, onehot_mat, ticksPerQuarterNote=1024):
        """

        :param midi_file_path: predict 후 만들어진 midi file 을 저장할 path
        :param onehot_mat: predict 결과 만들어진 notes 정보를 가진 matrix
        :param ticksPerQuarterNote: note 의 길이 기준
        :return:
        """
        # 틱 간격은 8분의 1음표 단위로
        tick_step = ticksPerQuarterNote / 2

        # onehot matrix에서 note 정보 추출.
        result = []
        for i in range(onehot_mat.shape[0]):
            for j in range(onehot_mat.shape[1]):
                # 새로운 노트가 시작될 경우에 리스트에 추가.
                if i == 0:
                    if onehot_mat[i, j] == 1:
                        result.append({'time': i * tick_step, 'note': j})
                else:
                    if onehot_mat[i, j] == 1 and onehot_mat[i - 1, j] == 0:
                        result.append({'time': i * tick_step, 'note': j})

        # unroll 패키지의 Keystrike를 이용해서 가공.
        ks = KeyStrikes(result)
        # 리스트 형태를 duration 을 가지도록 변환.
        ks = ks.quantized(ticksPerQuarterNote)
        # music21 스트림으로 변환.
        s = ks._to_music21stream()
        # 다시 midi 파일 형식으로 변환. ( ㅠ.ㅠ )
        mf = music21.midi.translate.streamToMidiFile(s)
        # 파일로 저장.
        mf.open(midi_file_path, 'wb')
        mf.write()
        mf.close()

        print '\tMidi file(=%s) successfully saved. Total 1/8 beats = %d' % (midi_file_path, onehot_mat.shape[0])

##
