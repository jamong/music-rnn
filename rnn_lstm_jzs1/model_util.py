# -*- coding: utf-8 -*-
import cPickle as pickle
import scipy.io as sio

import numpy as np
import theano
import theano.tensor as T


class ModelUtil(object):
    #
    # one-hot vector 모델을 위한 crossentropy objective_function
    #
    # one-hot vector를 만들어 곱하기 계산으로 cross entropy를 계산할 경우
    # 메모리가 많이 들어가므로 array indexing으로 cross entropy 를 계산함.
    #
    @staticmethod
    def one_hot_crossentropy(y_true, y_pred):

        # nan, inf 를 막기 위해 keras 에서 처리해주는 거 그대로 따왔음.
        if theano.config.floatX == 'float64':
            epsilon = 1.0e-9
        else:
            epsilon = 1.0e-7

        # 0 ~ 1 사이의 값으로 짤라줌.
        # ( 사실 softmax 가 0 ~ 1 값을 주므로 불필요하지만,
        # 혹시나 nan, inf 값이 나올까봐 처리하는 것 같음)
        y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # volabulary size
        voca_size = T.shape(y_pred)[-1]

        # indexing 을 위해 1D array 로 변환.
        y_pred = y_pred.flatten()
        y_true = y_true.flatten().astype('int32')

        # y_true의 word vector index를 1D array 에 맞게 변환.
        ix = T.arange(y_true.size) * voca_size + y_true

        # indexing instead of summation
        cce = -T.log(y_pred[ix])

        return cce

    @staticmethod
    def layer_info(model):

        print 'Checking model ...'

        ModelUtil._print_layer_shape(model)
        ModelUtil._print_layer_params(model)

    @staticmethod
    def _shape_to_str(shape):

        str = '('
        for i, n in enumerate(shape):
            if i > 0:
                str += ', {0}'.format(n)
        str += ')'
        return str

    @staticmethod
    def _print_layer_shape(model):

        print '\tlayer shapes'

        for i, layer in enumerate(model.layers):
            print '\t\t[%02d] %-30s %-30s %-30s' \
                  % (i, layer.__class__.__name__,
                     ModelUtil._shape_to_str(layer.input_shape),
                     ModelUtil._shape_to_str(layer.output_shape))

    @staticmethod
    def _print_layer_params(model):

        print '\tlayer parameters'

        total_params = 0
        for i, layer in enumerate(model.layers):
            layer_params = layer.count_params()
            total_params += layer_params

            print '\t\t[%02d] %-30s : %d' % \
                  (i, layer.__class__.__name__, layer_params)

        print '\t\t', '-' * 60
        print '\t\t%-35s : %d' % ('total', total_params)

    @staticmethod
    def node_info(graph):

        print 'Checking model ...'

        ModelUtil._print_node_shape(graph)
        ModelUtil._print_node_params(graph)

    @staticmethod
    def _print_node_shape(graph):

        print '\tnode shapes'

        for k, node in graph.nodes.iteritems():
            input_shape = ModelUtil._shape_to_str(node.input_shape)

            print '\t\t%-30s %-30s %-30s %-30s' \
                  % (k, node.__class__.__name__,
                     input_shape,
                     ModelUtil._shape_to_str(node.output_shape))

    @staticmethod
    def _print_node_params(graph):

        print '\tnode parameters'

        total_params = 0
        for k, node in graph.nodes.iteritems():
            node_params = node.count_params()
            total_params += node_params

            print '\t\t%-30s %-30s : %d' % \
                  (k, node.__class__.__name__, node_params)

        print '\t\t', '-' * 90
        print '\t\t%-61s : %d' % ('total', total_params)

    @staticmethod
    def save(path, *var_list):
        # 변수 저장.
        with open(path + '.pkl', 'w') as f:
            for v in var_list:
                pickle.dump(v, f)

    @staticmethod
    def load(path):
        # 변수 로딩 함수.
        def load_iter(path):
            with open(path + '.pkl', 'r') as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break

        # 리턴 리스트.
        ret_list = []
        for v in load_iter(path):
            ret_list.append(v)

        # 튜플로 리턴.
        if len(ret_list) == 1:
            return ret_list[0]
        else:
            return tuple(ret_list)

    @staticmethod
    def load_mat(path, *var_list, **kwargs):

        # 매트랩 벡터를 numpy 벡터로 변환할지 여부.
        mat2vec = True
        if 'mat2vec' in kwargs:
            mat2vec = kwargs['mat2vec']
        # 매트렙 벡터를 python 리스트로 변환할지 여부.
        mat2list = False
        if 'mat2list' in kwargs:
            mat2list = kwargs['mat2list']

        # 매트랩 파일 로딩.
        contents = sio.loadmat(path + '.mat')

        # return 할 변수 로딩.
        ret_list = []
        for v in var_list:
            try:
                value = contents[v]

                # 매트랩의 열벡터 행벡터를 numpy vector 로 변환.
                if type(value) == np.ndarray and value.ndim == 2 and \
                        (value.shape[0] == 1 or value.shape[1] == 1):
                    if mat2vec:
                        # 벡터로 변환.
                        value = value.flatten()
                    if mat2list:
                        # list 로 변환.
                        value = list(value)

                # list 에 추가.
                ret_list.append(value)
            except:
                ret_list.append(None)
                pass

        # tuple 로 리턴.
        if len(ret_list) > 1:
            return tuple(ret_list)
        else:
            return ret_list[0]

    @staticmethod
    def save_mat(path, **var_list):

        # 매트랩 파일에 저장하기.
        sio.savemat(path + '.mat', var_list)


class CodeUtil(object):
    UNICHR_START = 10  # 벡터코드 시작값 ( 0 ~ 9 까지는 본 시스템에서 예약함.)
    UNICHR_EOS = 0  # ascii null 을 EOS 로 사용함.
    UNICHR_OOC = 1  # Out of Character
    UNICHR_EMP = 2  # Empty mark
    UNICHR_RESERV1 = 3  # reserved
    UNICHR_RESERV2 = 4  # reserved
    UNICHR_RESERV3 = 5  # reserved

    # 바이트 당 dimension 수 정의.
    byte_size = 256

    @staticmethod
    def str_to_two_hot_vec(uni_str):

        ret = np.zeros((len(uni_str), CodeUtil.byte_size * 2), dtype='int32')
        for i, c in enumerate(uni_str):
            # 캐릭터 코드 읽기.
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # python 이 처리 가능한 유니코드 범위만 처리.
                ret[i, int(code / CodeUtil.byte_size) + CodeUtil.byte_size] = 1
                ret[i, code % CodeUtil.byte_size] = 1
            else:
                # 그 외의 것은 OOV 처리.
                ret[i, CodeUtil.byte_size] = 1
                ret[i, CodeUtil.UNICHR_OOC] = 1

        # numpy array 로 리턴.
        return np.array(ret)

    @staticmethod
    def str_to_seq_idx(uni_str):

        ret = np.zeros(len(uni_str) * 2, dtype='int32')
        for i, c in enumerate(uni_str):
            # 캐릭터 코드 읽기.
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # python 이 처리 가능한 유니코드 범위만 처리.
                ret[i * 2] = int(code / CodeUtil.byte_size)
                ret[i * 2 + 1] = code % CodeUtil.byte_size
            else:
                # 그 외의 것은 OOV 처리.
                ret[i * 2 + 1] = CodeUtil.UNICHR_OOC

        # numpy array 로 리턴.
        return np.array(ret)

    @staticmethod
    def str_to_seq_vec(uni_str):

        ret = np.zeros((len(uni_str) * 2, CodeUtil.byte_size), dtype='int32')
        for i, c in enumerate(uni_str):
            # 캐릭터 코드 읽기.
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # python 이 처리 가능한 유니코드 범위만 처리.
                ret[i * 2, int(code / CodeUtil.byte_size)] = 1
                ret[i * 2 + 1, code % CodeUtil.byte_size] = 1
            else:
                # 그 외의 것은 OOV 처리.
                ret[i * 2 + 1, CodeUtil.UNICHR_OOC]

        # numpy array 로 리턴.
        return np.array(ret)

    @staticmethod
    def seq_vec_to_str(vec):

        ret = []
        code = 0
        for i in range(vec.shape[0]):
            if i % 2 == 0:
                code = (vec[i].argmax()) * CodeUtil.byte_size
            else:
                code += vec[i].argmax()
                ret.append(unichr(code))
                code = 0

        # 문자열 리턴.
        return ''.join(ret)

    @staticmethod
    def idx_to_seq_vec(idx_mat, n_bytes):

        # 입력값을 일단 1차원으로 변경.
        x = idx_mat.flatten()

        # 바이트 단위로 짤라서 n-hot-vector 에 해당하는 index 를 생성함.
        idx_list = [np.mod(x, CodeUtil.byte_size ** (i + 1)) / CodeUtil.byte_size ** i for i in range(n_bytes)]
        # big endian 방식으로 변경
        idx_list = idx_list[::-1]

        # n-hot-vector 만들기.
        n_hot_mat = np.zeros((idx_mat.shape[0] * idx_mat.shape[1] * n_bytes, CodeUtil.byte_size),
                             dtype='int32')
        for i in range(len(idx_list)):
            for j in range(len(idx_list[i])):
                n_hot_mat[j * n_bytes + i, idx_list[i][j]] = 1

        # 원래 입력 데이터 형태로 변경.
        return n_hot_mat.reshape(idx_mat.shape[0], idx_mat.shape[1] * n_bytes, CodeUtil.byte_size)

    @staticmethod
    def seq_vec_to_idx(n_hot_mat, n_bytes):

        # 리턴할 인덱스 매트릭스 생성.
        cols = n_hot_mat.shape[1] / n_bytes
        idx_mat = np.zeros((n_hot_mat.shape[0], cols), dtype='int32')
        for i in range(cols):
            for j in range(n_bytes):
                # multiplier 곱해서 index 로 변경해줌.
                idx_mat[:, i] += n_hot_mat[:, i * n_bytes + j, :].argmax(axis=-1) * 256 ** (n_bytes - j - 1)

        return idx_mat
