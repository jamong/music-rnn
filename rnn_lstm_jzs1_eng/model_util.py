# -*- coding: utf-8 -*-
import cPickle as pickle
import scipy.io as sio

import numpy as np
import theano
import theano.tensor as T


class ModelUtil(object):
    #
    # cross-entropy objective function for one-hot vector model
    #
    # when calculate cross entrophy with multiplying after make one-hot vector,
    # capacity of using memory is so huge, so, calculate cross-entropy with array indexing
    #
    @staticmethod
    def one_hot_crossentropy(y_true, y_pred):

        # use Keras`s code to prevent nan, inf
        if theano.config.floatX == 'float64':
            epsilon = 1.0e-9
        else:
            epsilon = 1.0e-7

        # cut the values between 0 and 1
        # ( in fact, Softmax makes value 0 ~ 1, so this is not need,
        # i think, maybe, process this code to prevent unexpected nan, inf )
        y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= y_pred.sum(axis=-1, keepdims=True)

        # volabulary size
        voca_size = T.shape(y_pred)[-1]

        # convert to 1D array for indexing
        y_pred = y_pred.flatten()
        y_true = y_true.flatten().astype('int32')

        # change y_true`s word vector index to fit 1D array
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
        # save variables
        with open(path + '.pkl', 'w') as f:
            for v in var_list:
                pickle.dump(v, f)

    @staticmethod
    def load(path):
        # function to load variables
        def load_iter(path):
            with open(path + '.pkl', 'r') as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break

        # return list
        ret_list = []
        for v in load_iter(path):
            ret_list.append(v)

        # return type is tuple
        if len(ret_list) == 1:
            return ret_list[0]
        else:
            return tuple(ret_list)

    @staticmethod
    def load_mat(path, *var_list, **kwargs):

        # whether convert MatLab vector to NumPy vector or not
        mat2vec = True
        if 'mat2vec' in kwargs:
            mat2vec = kwargs['mat2vec']
        # whether convert MatLab vector to Python List or not
        mat2list = False
        if 'mat2list' in kwargs:
            mat2list = kwargs['mat2list']

        # load MatLab file
        contents = sio.loadmat(path + '.mat')

        # load variables to return
        ret_list = []
        for v in var_list:
            try:
                value = contents[v]

                # convert row-vector, column-vector of MatLab to NumPy vector
                if type(value) == np.ndarray and value.ndim == 2 and \
                        (value.shape[0] == 1 or value.shape[1] == 1):
                    if mat2vec:
                        # convert to vector
                        value = value.flatten()
                    if mat2list:
                        # convert to list
                        value = list(value)

                # append to list
                ret_list.append(value)
            except:
                ret_list.append(None)
                pass

        # return as tuple
        if len(ret_list) > 1:
            return tuple(ret_list)
        else:
            return ret_list[0]

    @staticmethod
    def save_mat(path, **var_list):

        # save as MatLab format
        sio.savemat(path + '.mat', var_list)


class CodeUtil(object):
    UNICHR_START = 10  # start number of vector code (0~9 is reserved by System)
    UNICHR_EOS = 0  # use ascii null as EOS
    UNICHR_OOC = 1  # Out of Character
    UNICHR_EMP = 2  # Empty mark
    UNICHR_RESERV1 = 3  # reserved
    UNICHR_RESERV2 = 4  # reserved
    UNICHR_RESERV3 = 5  # reserved

    # define dimension per byte
    byte_size = 256

    @staticmethod
    def str_to_two_hot_vec(uni_str):

        ret = np.zeros((len(uni_str), CodeUtil.byte_size * 2), dtype='int32')
        for i, c in enumerate(uni_str):
            # read character code
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # process unicode what can be treated by python
                ret[i, int(code / CodeUtil.byte_size) + CodeUtil.byte_size] = 1
                ret[i, code % CodeUtil.byte_size] = 1
            else:
                # treat others as OOV ( Out of Value )
                ret[i, CodeUtil.byte_size] = 1
                ret[i, CodeUtil.UNICHR_OOC] = 1

        # return by numpy array
        return np.array(ret)

    @staticmethod
    def str_to_seq_idx(uni_str):

        ret = np.zeros(len(uni_str) * 2, dtype='int32')
        for i, c in enumerate(uni_str):
            # read character code
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # process unicode what can be treated by python
                ret[i * 2] = int(code / CodeUtil.byte_size)
                ret[i * 2 + 1] = code % CodeUtil.byte_size
            else:
                # treat others as OOV
                ret[i * 2 + 1] = CodeUtil.UNICHR_OOC

        # return by numpy array
        return np.array(ret)

    @staticmethod
    def str_to_seq_vec(uni_str):

        ret = np.zeros((len(uni_str) * 2, CodeUtil.byte_size), dtype='int32')
        for i, c in enumerate(uni_str):
            # read character code
            code = ord(c)
            if CodeUtil.UNICHR_START <= code < 2 ** 16:
                # process unicode what can be treated by python
                ret[i * 2, int(code / CodeUtil.byte_size)] = 1
                ret[i * 2 + 1, code % CodeUtil.byte_size] = 1
            else:
                # treat others as OOV
                ret[i * 2 + 1, CodeUtil.UNICHR_OOC]

        # return by numpy array
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

        # return string
        return ''.join(ret)

    @staticmethod
    def idx_to_seq_vec(idx_mat, n_bytes):

        # convert dimension of input to 1D
        x = idx_mat.flatten()

        # create index of n-hot-vector with cutting by byte
        idx_list = [np.mod(x, CodeUtil.byte_size ** (i + 1)) / CodeUtil.byte_size ** i for i in range(n_bytes)]
        # conver to big endian format
        idx_list = idx_list[::-1]

        # make n-hot-vector
        n_hot_mat = np.zeros((idx_mat.shape[0] * idx_mat.shape[1] * n_bytes, CodeUtil.byte_size),
                             dtype='int32')
        for i in range(len(idx_list)):
            for j in range(len(idx_list[i])):
                n_hot_mat[j * n_bytes + i, idx_list[i][j]] = 1

        # convert to original input data format
        return n_hot_mat.reshape(idx_mat.shape[0], idx_mat.shape[1] * n_bytes, CodeUtil.byte_size)

    @staticmethod
    def seq_vec_to_idx(n_hot_mat, n_bytes):

        # declare index matrix to return
        cols = n_hot_mat.shape[1] / n_bytes
        idx_mat = np.zeros((n_hot_mat.shape[0], cols), dtype='int32')
        for i in range(cols):
            for j in range(n_bytes):
                # convert to index by multiply with multiplier
                idx_mat[:, i] += n_hot_mat[:, i * n_bytes + j, :].argmax(axis=-1) * 256 ** (n_bytes - j - 1)

        return idx_mat
