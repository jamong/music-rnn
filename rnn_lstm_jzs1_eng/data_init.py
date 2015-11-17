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
        # criterion to apply indexing to samples
        # after converting midi files, tick_step is changed with Quater note
        # tick_step = mf.ticksPerQuarterNote / 2            # formula to calculate tick_setp
        # although tick_step is initialized with 512, actually this is re-initialized when program loads midi files
        self.tick_step = 512

        self.samples_num = 0                                # samples number, will initialized
        self.seq_length = 0                                 # sequence length, will initialized
        # hidden size
        self.note_dim = 128                                 # midi file`s standard : a total of 128 notes
        self.data_dim = self.note_dim + 1                   # add EOS
        self.hidden_size = self.data_dim                    # just naming, data dimention == hidden size


    def get_data_set_from_midi(self, path_data, target_str=""):
        """
        .
        :param path_data: directory path information what is including midi files
        :param target_str: string which is included with target files will be loaded from directory path
        :return: X train data set, Y train data set, samples number, sequence length, hidden size
        """
        # get target samples list
        self.samples_num, samples_list = self.get_samples_list(path_data, target_str)

        # get longest sequence length
        self.seq_length = self.get_data_set_max_seq_length(path_data, samples_list)

        # for debug
        print "\tdebug --------> self.seq_length : {0}\n\n".format(self.seq_length)

        # init data_set
        X_train = np.zeros((self.samples_num, self.seq_length, self.hidden_size), dtype=int)
        Y_train = np.copy(X_train)

        #
        # convert data of each file to one-hot vector, and save one-hot vector with X_train
        #

        samples_cnt = 0                                         # set sample`s start number : which midi file is
        for filename in samples_list:
            one_X_train = self.load_midi(path_data, filename)   # get notes info of one midi file

            # set matrix what is containing extracted notes data of one midi file to X_train(train data set)
            #
            # left zero-padding
            X_train[samples_cnt][self.seq_length - one_X_train.shape[0] + 1:][:] = np.copy(one_X_train[:-1][:])

            samples_cnt += 1

        # set Y_train, precede one index than X_train
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
        find midi file having longest play time among the data files to be loaded,
        calculate the midi files notes end index.
        it will be the criterion of sequence length
        :param path_data:
        :param samples_list:
        :return: notes index number which will be the criterion of seq_length
        """
        # for debug
        whatislongestfile = ""
        print "\n...... get info : longest sequence length ......"
        max_ks_time = 0
        mfticksperquarternote = 0
        for filename in samples_list:
            ks, mfticksperquarternote = self.load_midi(path_data, filename, get_longest_length=True)

            # get sequence length of file having longest play time
            now_ks_time = ks.keystrikes[-1]['time'] + ks.keystrikes[-1]['duration'] * mfticksperquarternote
            if max_ks_time < now_ks_time:
                max_ks_time = now_ks_time
                whatislongestfile = filename

        tick_step = mfticksperquarternote / 2             # tick step is 8th note, quaver
        longest_seq_length = max_ks_time / tick_step + 1  # add 1 index for EOS

        print "\n\tLongest File name : {0}, sequence length : {1}\n\n".format(whatislongestfile, longest_seq_length)
        return longest_seq_length



    def get_samples_list(self, path_data, target_str=""):
        """
        receive the path of midi files and string of target files,
        then, return the number and names of files for training
        :param path_data: path including midi files
        :param target_str: string which is included target files
        :return: midi files number, file names list
        """
        print "\n...... get samples list ......"
        # file list at path
        samples_list = os.listdir(path_data)

        # extract file`s name containing target_str
        if target_str != "":
            temp_list_train_files = []
            for item in samples_list:
                if target_str in item:
                    temp_list_train_files.append(item)
            samples_list = temp_list_train_files

        # sort
        samples_list.sort()
        # init sample number
        self.samples_num = len(samples_list)

        print "\n\tReturn {0} file name list.".format(len(samples_list))
        return len(samples_list), samples_list


    def load_midi(self, path_data, filename, get_longest_length=False):
        """
        .
        :param path_data: path including midi files
        :param filename: midi file name
        :param get_longest_length: Flag to get only the length of files having longest play time, or not
        :return:
        """
        # read midi file
        mf = music21.midi.MidiFile()
        path_midifile = path_data + '/' + filename
        mf.open(path_midifile)
        mf.read()
        mf.close()

        # midi -> music21 stream -> midi
        # change criterion of ticksperQuarterNote of mf
        s = music21.midi.translate.midiFileToStream(mf)
        mf = music21.midi.translate.streamToMidiFile(s)

        #
        # get notes infomation while loop tracks
        #
        #   track`s event structure
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

        # use unroll.KeyStrikes
        ks = KeyStrikes(result)
        # convert list to have duration
        ks = ks.quantized(mf.ticksPerQuarterNote)

        # use If u wanna know the sequence length of file having longest play time,
        # set get_longest_length = True, then this function will return
        # the longest sequence length and mf.ticksPerQuarterNote 만 반환.
        if get_longest_length:
            return ks, mf.ticksPerQuarterNote

        #
        # conver one-hot vector format
        #

        # tick step = 8th note, quaver
        self.tick_step = mf.ticksPerQuarterNote / 2
        # total tick count
        total_tick = ks.keystrikes[-1]['time'] + ks.keystrikes[-1]['duration'] * mf.ticksPerQuarterNote
        # total sequence number
        total_beat = total_tick / self.tick_step

        # temporary matrix to save one hot vector of one midi file
        one_X_train = np.zeros((total_beat + 1, self.data_dim), dtype='int32')

        # decoding the notes info while loop with having 1024 criterion
        for strike in ks.keystrikes:
            # index, criterion = 8th note(quaver)
            idx = strike['time'] / self.tick_step
            # during time of note, criterion = 8th note(quaver), calculate how many indices will be filled
            dur = int(strike['duration'] * 2)

            # set one-hot vector by note
            for note in strike['note']:
                # set 1 to index as many as duration of note
                for i in range(dur):
                    one_X_train[idx + i, note] = 1

        # add EOS at end of sequence, EOS is case that data_dim[-1] = 1
        one_X_train[-1, self.note_dim] = 1

        print '\tMidi file(=%s) successfully loaded. Total 1/8 beats = %d' % (path_midifile, total_beat)

        return one_X_train


    # do not use in this traing model, just reserve function
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

        :param midi_file_path: path to save midi file which is made after predicting
        :param onehot_mat: matrix having notes infomation made after predicting
        :param ticksPerQuarterNote: criterion of note`s length(duration)
        :return:
        """
        # tick step = 8th note, quaver
        tick_step = ticksPerQuarterNote / 2

        # extract note info from one-hot matrix
        result = []
        for i in range(onehot_mat.shape[0]):
            for j in range(onehot_mat.shape[1]):
                # if flow is started with new note info, then add new info to result list
                if i == 0:
                    if onehot_mat[i, j] == 1:
                        result.append({'time': i * tick_step, 'note': j})
                else:
                    if onehot_mat[i, j] == 1 and onehot_mat[i - 1, j] == 0:
                        result.append({'time': i * tick_step, 'note': j})

        # manufacture result list with unroll package`s Keystrike
        ks = KeyStrikes(result)
        # convert ks to have duration
        ks = ks.quantized(ticksPerQuarterNote)
        # convert ks to music21`s stream
        s = ks._to_music21stream()
        # convert s to midi format, again T^T
        mf = music21.midi.translate.streamToMidiFile(s)
        # save mf info with file
        mf.open(midi_file_path, 'wb')
        mf.write()
        mf.close()

        print '\tMidi file(=%s) successfully saved. Total 1/8 beats = %d' % (midi_file_path, onehot_mat.shape[0])

##
