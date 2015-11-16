# -*- coding: utf-8 -*-
import music21
import numpy as np
from unroll.KeyStrikes import KeyStrikes


class MidiUtil(object):
    @staticmethod
    def load_midi(midi_file_path):

        # 미디 파일 읽
        mf = music21.midi.MidiFile()
        mf.open(midi_file_path)
        mf.read()
        mf.close()

        # music21 stream 으로 변환했다가 다시 midi 형태로 변환.
        # (시간처리 등 전처리 알아서 하게...)
        s = music21.midi.translate.midiFileToStream(mf)
        mf = music21.midi.translate.streamToMidiFile(s)

        # 각 트랙별로 루프돌면서 note 정보 읽어 오기.
        result = []
        for i in range(len(mf.tracks)):
            t = 0
            # 트랙의 event 를 추적함.
            for e in mf.tracks[i].events:
                # 시간 이벤트일 경우 시간 업데이트
                if e.isDeltaTime() and e.time is not None:
                    t += e.time  # 델타 tick count 라서 누적해줌.
                # 키가 눌려진 경우에는 list에 정보 추가.
                elif e.isNoteOn() and e.pitch is not None and e.velocity != 0:
                    result.append({'time': t, 'note': e.pitch})

        # unroll 패키지의 Keystrike를 이용해서 가공.
        ks = KeyStrikes(result)
        # 리스트 형태를 duration 을 가지도록 변환.
        ks = ks.quantized(mf.ticksPerQuarterNote)

        #
        # one-hot vector 형식으로 변환.
        #

        # 틱 간격은 8분의 1음표 단위로
        tick_step = mf.ticksPerQuarterNote / 2
        # 총 틱 카운트는 맨 마지막 tick count 에다가 마지막 note 의 지속시간을 더해주면 됨.
        total_tick = ks.keystrikes[-1]['time'] + ks.keystrikes[-1]['duration'] * mf.ticksPerQuarterNote
        # 총 1/8 음표 갯수( 8분의 1음표 단위로 몇 개인지...)
        total_beat = total_tick / tick_step

        # midi 파일 규격에 의하면 총 128 음표까지 가능.
        note_dim = 128
        data_dim = note_dim + 1  # 음표 갯수에 EOS 한개 추가.

        # one hot vector 를 저장 할 matrix.
        notes = np.zeros((total_beat + 1, data_dim), dtype='int32')

        # 눌려진 건반 정보를 루프 돌며 디코딩.
        for strike in ks.keystrikes:

            # 1/8 음표 단위의 index
            idx = strike['time'] / tick_step
            # 1/8 음표 단위의 지속시간
            dur = int(strike['duration'] * 2)

            # note 별로 one-hot vector 채우기.
            for note in strike['note']:
                # 지속시간 만큼 marking
                for i in range(dur):
                    notes[idx + i, note] = 1

        # 끝에 EOS 추가. ( EOS 는 128 th column 이 1인 경우로 정의함. )
        notes[-1, note_dim] = 1

        print '\tMidi file(=%s) successfully loaded. Total 1/8 beats = %d' % (midi_file_path, total_beat)

        # 결과값 리턴.
        return notes

    @staticmethod
    def save_midi(midi_file_path, onehot_mat, ticksPerQuarterNote=1024):

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


# 테스트 코드.
if __name__ == '__main__':
    # 미디 파일을 읽어서 one-hot matrix로 변환.
    ov = MidiUtil.load_midi('reels_simple_chords_1.mid')
    # 결과 출력.
    print 'Loaded one-hot matrix\'s # of chords for each beat are \n', ov.sum(axis=-1)
    # one-hot matrix를 미디 파일로 저장.
    MidiUtil.save_midi('example.mid', ov)
