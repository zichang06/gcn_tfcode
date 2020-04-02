from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# 定义基类 Layer
# 属性：name (String) => 定义了变量范围；logging (Boolean) => 打开或关闭TensorFlow直方图日志记录
# 方法：init()(初始化)，_call()(定义计算)，call()(调用_call()函数)，_log_vars()
# 定义Dense Layer类，继承自Layer类
# 定义GraphConvolution类，继承自Layer类。

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


# 有点像计数，该名字的layer有几个了，没有的话，返回1，有的话+1再返回
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

# 稀疏矩阵的dropout操作
# noise_shape在调用时的参数是：num_features_nonzero - features[1].shape，矩阵中非零元素的个数
# features[1].shape: (49216,); features[0].shape: (49216,2), type(features[1].shape):tuple
# 参考 https://www.jb51.net/article/182591.htm  - tf.nn.dropout解析，进而分析sparse_dropout
# 有时可以研读下原型函数的源码，以辅助理解

# 参数x和keep_prob与tf.nn.dropout一致，noise_shape为x中非空元素的个数，
# 如果x中有4个非空值，则noise_shape为[4]，keep_tensor的元素为[keep_prob, 1.0 + keep_prob)的均匀分布，
# 通过tf.floor向下取整得到标记张量drop_mask，tf.sparse_retain用于在一个 SparseTensor 中保留指定的非空值。
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob  # 比如0.9
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    # https://www.w3cschool.cn/tensorflow_python/tensorflow_python-ad3z2lx3.html
    # 在一个 SparseTensor(即x)中保留指定的非空值.  tf.sparse...可以看下↑↑,稀疏矩阵的表示
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    # _call()(定义计算)
    # 这个_call是恒等映射，可以看到下面子类override这个函数，且有具体的计算
    def _call(self, inputs):
        return inputs

    # __call__()(调用_call()函数)   
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)  # 记录该层输入特征值
            outputs = self._call(inputs)  
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)  # 记录该层输出特征值
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act  # 激活函数
        self.sparse_inputs = sparse_inputs   # 是否是稀疏数据 
        self.featureless = featureless   # 输入的数据带不带特征矩阵
        self.bias = bias  # 是否有偏置

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    #重写了_call 函数，其中对稀疏矩阵做 drop_out:sparse_dropout()
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)  

        # transform
        # 呃啊，其实就是单纯的MLP全连接，x的各列就是通道，然后映射...
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

#从 Layer 继承下来得到图卷积网络，
# 与denseNet的唯一差别是_call函数和__init__函数（self.support = placeholders['support']的初始化）
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless  # 输入的数据带不带特征矩阵
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # k阶切比雪夫实现。参考论文式(5)
        # 矩阵形式，每一阶截断参考式(8)实现
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],  # X*W
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]  # W 或言之 I*W
            support = dot(self.support[i], pre_sup, sparse=True) # Tk(L)*X*W 
            supports.append(support)
        output = tf.add_n(supports)   # 切比雪夫各阶截断累加起来。

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
