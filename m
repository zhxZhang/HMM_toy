
initializations

import numpy as np
import tensorflow as tf
import ast
_FLOATX = "float32"


def tf_variable(value, dtype=_FLOATX, name=None):
    """
    Instantiates a tensor.
    # Arguments
         value: numpy array, initial value of the tensor.
         dtype: tensor type.
         name: optional name string for the tensor.

    # Returns
        Tensor variable instance.
    """
    return tf.get_variable(initializer=np.asarray(value, dtype=dtype), name=name)


def get_fans(shape):
    """
      Calculate the fan-in and fan-out of shape
      # Returns
        the fan-in and fan-out of shape
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        # assuming convolution kernels (2D or 3D).
        # TF kernel shape: (..., input_depth, depth)
        fan_in = np.prod(shape[:-1])
        fan_out = shape[-1]
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(shape, scale=0.05, seed=None, name=None):
    """
      A wrapper for np.random.uniform
      # Returns
        generate tensor variable from uniform distribution
    """
    np.random.seed(seed)
    return tf_variable(np.random.uniform(low=-scale, high=scale, size=shape),
                      name=name)


def normal(shape, scale=0.05, seed=None, name=None):
    """
      A wrapper for np.random.normal
      # Returns
       generate tensor variable from normal distribution
    """
    np.random.seed(seed)
    return tf_variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)


def lecun_uniform(shape, seed=None, name=None):
    """
       Generate tensor variable from lecun_uniform distribution
       Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale, name=name)


def glorot_normal(shape, seed=None, name=None):
    """
       Generate tensor variable from glorot_normal distribution
       Reference: Glorot & Bengio, AISTATS 2010
    """
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, name=name)


def glorot_uniform(shape, seed=None, name=None):
    """
       Generate tensor variable from glorot_uniform distribution
       Reference: Glorot & Bengio, AISTATS 2010
    """
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def he_normal(shape, seed=None, name=None):
    """
    Generate tensor variable from he_normal distribution
    Reference:  He et al., http://arxiv.org/abs/1502.01852
    """
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def he_uniform(shape, seed=None, name=None):
    """
    Generate tensor variable from he_uniform distribution
    Reference:  He et al., http://arxiv.org/abs/1502.01852
    """
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, name=name)


def zero(shape, seed=None, name=None):
    """
    Generate tensor variable initialized by zeros
    """
    return tf.get_variable(initializer=tf.constant_initializer(0.), shape=shape, name=name)


def one(shape, seed=None, name=None):
    """
    Generate tensor variable initialized by ones
    """
    return tf.get_variable(initializer=tf.constant_initializer(1.), shape=shape, name=name)

layers
import tensorflow as tf
import sys, os
import initializations as tf_init
import numpy as np
import ast
_EPSILON = 10e-8
_FLOATX = "float32"

FLAGS = tf.app.flags.FLAGS
# moving average decay factor
tf.app.flags.DEFINE_float('bn_stats_decay_factor', 0.99,
                          "moving average decay factor for stats on batch normalization")


def lrelu(x, a=0.1):
    """
    Define lrelu activation function
    """
    if a < 1e-16:
        return tf.nn.relu(x)
    else:
        return tf.maximum(x, a * x)


def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn"):
    """
    batch normalization
    """
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    mean = tf.reduce_mean(x, axis)
    var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
    avg_mean = tf.get_variable(
        name=name + "_mean",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
        trainable=False
    )

    avg_var = tf.get_variable(
        name=name + "_var",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections,
        trainable=False
    )
    # the variable that controls the batch normalization
    gamma = tf.get_variable(
        name=name + "_gamma",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections
    )

    beta = tf.get_variable(
        name=name + "_beta",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
    )

    if is_training:
        avg_mean_assign_op = tf.no_op()
        avg_var_assign_op = tf.no_op()
        if update_batch_stats:
            avg_mean_assign_op = tf.assign(
                avg_mean,
                FLAGS.bn_stats_decay_factor * avg_mean + (1 - FLAGS.bn_stats_decay_factor) * mean)
            avg_var_assign_op = tf.assign(
                avg_var,
                FLAGS.bn_stats_decay_factor * avg_var + (n / (n - 1))
                * (1 - FLAGS.bn_stats_decay_factor) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            z = (x - mean) / tf.sqrt(1e-6 + var)
    else:
        z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

    return gamma * z + beta


def fc(x, dim_in, dim_out, seed=None, name='fc'):
    """
    Fully collected layer
    """
    num_units_in = dim_in
    num_units_out = dim_out
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    # define the weights w of fully collected layer
    weights = tf.get_variable(name + '_W',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer)
    # define the biases b of fully collected layer
    biases = tf.get_variable(name + '_b',
                             shape=[num_units_out],
                             initializer=tf.constant_initializer(0.0))
    # wx+b
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False, seed=None, name='conv'):
    """
    Convolution layer
    """
    # filter size
    shape = [ksize, ksize, f_in, f_out]
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    weights = tf.get_variable(name + '_W',
                            shape=shape,
                            dtype='float',
                            initializer=initializer)
    # convolution operation
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)

    # whether to add bias
    if use_bias:
        bias = tf.get_variable(name + '_b',
                               shape=[f_out],
                               dtype='float',
                               initializer=tf.zeros_initializer)
        return tf.nn.bias_add(x, bias)
    else:
        return x


def conv1d(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False, seed=None, name='conv'):
    """
    Convolution along one dimension
    """
    shape = [ksize, x.shape[2], f_in, f_out]
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
    weights = tf.get_variable(name + '_W',
                            shape=shape,
                            dtype='float',
                            initializer=initializer)
    # convolution operation
    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)

    # whether to add bias
    if use_bias:
        bias = tf.get_variable(name + '_b',
                               shape=[f_out],
                               dtype='float',
                               initializer=tf.zeros_initializer)
        return tf.nn.bias_add(x, bias)
    else:
        return x


def avg_pool(x, ksize=2, stride=2):
    """
    average pooling layer
    """
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def max_pool(x, ksize=2, stride=2):
    """
    max pooling layer
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def word_dropout(x, dropout_keep_prob, seed = None):
    """
    word dropout layer by keeping dropout_keep_prob
    """
    rng = np.random.RandomState(1234)
    sequence_length = int(x.shape[1])
    # dropout_keep_prob is the probability to keep
    if dropout_keep_prob < 1.:
        return tf.nn.dropout(x, dropout_keep_prob, noise_shape=[1, sequence_length, 1], seed=rng.randint(123456))
    return x


def ce_loss(logit, y):
    """
    cross entropy loss
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


def accuracy(logit, y):
    """
    Compute the accuracy of prediction
    """
    # prediction result
    pred = tf.argmax(logit, 1)
    # ground-truth
    true = tf.argmax(y, 1)
    return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))


def logsoftmax(x):
    """
    Compute the logsoftmax
    """
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit, include_ent_term=False):
    """
    Compute the KL-divergence with logit
    """
    # logit is the value before feeding into softmax
    q = tf.nn.softmax(q_logit)
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    # whether to include entropy term
    if(include_ent_term):
        qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
        return qlogq - qlogp
    else:
        return -qlogp


def kl_divergence(q_y, p_y, main_obj_type = 'CE', include_ent_term=False):
    """
     Compute the KL-divergence according to whether it is cross entropy or quadratic entropy
    """
    # cross entropy
    if(main_obj_type=='CE'):
        p_y = tf.clip_by_value(p_y, tf.cast(_EPSILON, dtype=_FLOATX),
                                    tf.cast(1. - _EPSILON, dtype=_FLOATX))
        # whether to include entropy term
        if(include_ent_term):
            q_y = tf.clip_by_value(q_y, tf.cast(_EPSILON, dtype=_FLOATX),
                                        tf.cast(1. - _EPSILON, dtype=_FLOATX))
            return tf.reduce_mean(tf.reduce_sum(q_y * (tf.log(q_y) - tf.log(p_y)), 1))
        else:
            return - tf.reduce_mean(tf.reduce_sum(q_y * tf.log(p_y), 1))
    # quadratic entropy
    elif(main_obj_type=='QE'):
        return tf.reduce_mean(tf.reduce_sum((p_y-q_y)**2, 1))
    else:
        raise NotImplementedError()


def entropy_y_x(logit):
    """
     Define entropy based on logit
    """
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))


def categorical_crossentropy(logits, labels):
    """
     Define categorical cross entropy based on logits and labels
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def _variable_on_cpu(name, shape, initializer):
    """
     Define variable on cpu
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var    

losses
import tensorflow as tf
import numpy
import sys, os
import layers as L

aggregation_method = tf.AggregationMethod.DEFAULT
#aggregation_method = tf.AggregationMethod.ADD_N
#aggregation_method = 2


def get_normalized_vector(d):
    """
    Perform normalization for vector d
    """
    d /= (1e-12 + tf.reduce_max(tf.abs(d), [1], keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), [1], keep_dims=True))
    return d


def get_perturbation(d, epsilon, norm_constraint = 'L2'):
    """
    Get perturbation on data
    """
    if (norm_constraint == 'max'):
        return epsilon * tf.sign(d)
    elif (norm_constraint == 'L2'):
        return epsilon * get_normalized_vector(d)
    else:
        raise NotImplementedError()


def virtual_adversarial_loss(x, forward_func, epsilon, num_power_iter=1, xi=1e-6, is_training=True, name="vat_loss"):
    """
    Compute virtual adversarial loss
    See paper "Adversarial training methods for semi-supervised text classification"
    """
    y_p = forward_func(x)
    d = tf.random_normal(shape=tf.shape(x))
    for _ in range(num_power_iter):
        d = xi * get_normalized_vector(d)
        y_m = forward_func(x + d, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(y_p, y_m)
        grad = tf.gradients(dist, [d], aggregation_method=aggregation_method)[0] / xi
        d = tf.stop_gradient(grad)
    r_vadv = get_perturbation(d, epsilon, norm_constraint = 'L2') # get perturbation
    y_p_hat = tf.stop_gradient(y_p)
    y_m = forward_func(x + r_vadv, is_training=is_training, update_batch_stats=False)
    loss = L.kl_divergence_with_logit(y_p_hat, y_m)
    return tf.identity(loss, name=name)


def adversarial_loss(x, target, forward_func, epsilon, is_training=True, name="at_loss"):
    """
    Compute adversarial loss
    See paper "Adversarial training methods for semi-supervised text classification"
    """
    nll_cost = L.categorical_crossentropy(forward_func(x), target)
    grad = tf.gradients(nll_cost, [x], aggregation_method=aggregation_method)[0]
    grad = tf.stop_gradient(grad)
    r_adv = get_perturbation(grad, epsilon, norm_constraint = 'L2') # get perturbation
    y = forward_func(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.categorical_crossentropy(y, target)
    return tf.identity(loss, name=name)

model
import tensorflow as tf
import numpy as np
import layers as L
import losses as costs
import initializations as tf_init


class DeepModel(object):
    def __init__(self, 
                 input_x, 
                 input_y, 
                 input_ul_x, 
                 params, 
                 dropout_keep_prob = 1., 
                 word_dropout_keep_prob = 1., 
                 init_w2v = None, 
                 seed = None,
                 reuse_lstm=None,
                 is_training=False):
        """
        Initialization function
        # Arguments
         input_x: input data
         input_y: input label
         input_ul_x: unsupervised learning data,ie, unlabelled data
         params: parameters
         dropout_keep_prob: 1-dropout_rate
         word_dropout_keep_prob: input word dropout
         init_w2v: initialized word vector
         seed: random seed
         reuse_lstm: whether to reuse lstm
         is_training: whether it is in training stage
        """

        self.params = params
        # generate random state
        if seed is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self.params.init_weights_seed)

        # the maximum length of a document,ie, the sequence length
        sequence_length = params.doc_maxlen
        # the size of word embedding vector
        embedding_size = params.embedding_size
        # the size of vocabulary
        vocab_size = params.vocab_len
        # the predefined number of classes
        num_classes = params.num_classes
         
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.word_dropout_keep_prob = word_dropout_keep_prob

        self.reuse_lstm = reuse_lstm
        # the size of lstm state vector
        self.lstm_hidden_units = params.lstm_hidden_units
        # the size of hidden units
        self.softmax_hidden_units = params.softmax_hidden_units
 
        # word embeding
        with tf.variable_scope("embedding"):
            # using pre-trained word vector
            if init_w2v is not None:
                initializer = tf.constant(init_w2v)
                W = tf.get_variable(initializer=initializer, name="W")
            # random initialized word vector
            else:
                # initializer = tf.contrib.layers.variance_scaling_initializer(seed=self.rng.randint(123456))
                initializer = tf.random_uniform_initializer(-1.0, 1.0)
                W = tf.get_variable(initializer=initializer, shape=[vocab_size, embedding_size], name="W")
            # apply word dropout to word embedding
            input_e = L.word_dropout(tf.nn.embedding_lookup(W, input_x), word_dropout_keep_prob, seed=self.rng.randint(123456))
            # apply word dropout to unlabelled data
            if input_ul_x is not None:
                input_ul_e = L.word_dropout(tf.nn.embedding_lookup(W, input_ul_x), word_dropout_keep_prob, seed=self.rng.randint(123456))

        # base network
        # whether to use cnn model or bilstm model
        if params.model_method == 'cnn':
            self.forward_train = self.cnn_forward
        elif params.model_method == 'bilstm':
            self.forward_train = self.bilstm_forward
    
        # Calculate Mean cross-entropy loss
        logits = self.forward_train(input_e, add_weight_decay=self.is_training) 
        nll_loss = L.categorical_crossentropy(logits, input_y)
        
        additional_loss = 0
        # is_training = False will not call to build addtional model in validation and test phase.
        if self.is_training:
            self.reuse_lstm = True
            scope = tf.get_variable_scope()
            scope.reuse_variables()

            # build base cnn or bilstm model, no additional loss is needed
            if self.params.cost_type.upper() == "BASE":
                additional_loss = 0
            # build Adversarial Training model, adversarial_loss is needed
            elif self.params.cost_type.upper() == "AT":
                additional_loss = costs.adversarial_loss(input_e, input_y, self.forward_train,
                                                                 self.params.vat_epsilon)
            # build Virtual Adversarial Training model, virtual_adversarial_loss is needed
            elif self.params.cost_type.upper() == "VAT":
                additional_loss = costs.virtual_adversarial_loss(input_ul_e, self.forward_train,
                                                                 self.params.vat_epsilon)
            # build Virtual Adversarial Training model, plus entropy loss
            elif self.params.cost_type.upper() == "VATENT":
                additional_loss = costs.virtual_adversarial_loss(input_ul_e, self.forward_train,
                                                                 self.params.vat_epsilon)
                ent_loss = L.entropy_y_x(self.forward_train(input_ul_e))
                additional_loss +=  ent_loss

        with tf.variable_scope("loss"):
            # In training stage, weight_decay loss is added
            if self.is_training:
                l2_loss = tf.add_n(tf.get_collection('weight_decays'), name='wd_loss')
            else:
                l2_loss = 0
 
            nll_loss = tf.identity(nll_loss, name = 'nll_loss')
            # the total loss
            self.loss = tf.identity(nll_loss + params.vat_lambda*additional_loss + params.l2_reg_lambda * l2_loss, name='total_loss')
        #  the prediction value
        with tf.variable_scope("output"):
            self.scores = tf.nn.softmax(logits, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Accuracy
        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(input_y, 1), name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    # def cnn forward network
    def cnn_forward(self, x, is_training=True, update_batch_stats=False, add_weight_decay=False):
        # Note that this random state is used to keep the dropout out settings the same in AT and VAT training.
        dropout_rng = np.random.RandomState(1234)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, -1)

        filter_sizes = [3,4,5]
        # for each filter size, there are num_filters filters respectively
        num_filters = 128
        # the final feature size before classification
        feature_size = self.softmax_hidden_units
        sequence_length, embedding_size = int(x.shape[1]), int(x.shape[2])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]

                # initializer = tf.contrib.layers.variance_scaling_initializer(seed=self.rng.randint(123456))
                initializer = tf.truncated_normal_initializer(stddev=0.1, seed=dropout_rng.randint(123456)) 
                W = tf.get_variable(initializer=initializer, shape=filter_shape, name="W")
                b = tf.get_variable(initializer=tf.constant_initializer(0.), shape=[num_filters], name="b")
                # W = tf_init.get_init(init = 'uniform', shape=filter_shape, name = "W")
                # b = tf_init.get_init(init = 'zero', shape=[num_filters], name = "b")

                # convolution operation
                conv = tf.nn.conv2d(x,  W,
                       strides=[1, 1, 1, 1],
                       padding="VALID",
                       name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # a feature-map will generate only one maximum value
                pooled = tf.nn.max_pool(
                         h,
                         ksize=[1, sequence_length - filter_size + 1, 1, 1],
                         strides=[1, 1, 1, 1],
                         padding='VALID',
                         name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        # reshape the pooled features into a vector
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Final (unnormalized) scores
        with tf.variable_scope("cnn-output"):
            # apply dropout to the pooled features
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob, seed=dropout_rng.randint(123456))

            # initializer = tf.contrib.layers.variance_scaling_initializer(seed=self.rng.randint(123456))
            initializer = tf.truncated_normal_initializer(stddev=0.1, seed=self.rng.randint(123456)) 
            W0 = tf.get_variable(initializer=initializer, shape=[num_filters_total, feature_size], name="W0")
            b0 = tf.get_variable(initializer=tf.constant_initializer(0.), shape=[feature_size], name="b0")

            # dropout after fully collected layer
            h_drop = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(h_drop, W0, b0)), self.dropout_keep_prob, seed=dropout_rng.randint(123456))

            # initializer = tf.contrib.layers.variance_scaling_initializer(seed=self.rng.randint(123456))
            initializer = tf.truncated_normal_initializer(stddev=0.1, seed=self.rng.randint(123456)) 
            W = tf.get_variable(initializer=initializer, shape=[feature_size, self.params.num_classes], name="W")
            b = tf.get_variable(initializer=tf.constant_initializer(0.), shape=[self.params.num_classes], name="b")
            # add weight decay for the two fully collected layer weights
            if add_weight_decay:
                tf.add_to_collection('weight_decays', tf.nn.l2_loss(W0))
                tf.add_to_collection('weight_decays', tf.nn.l2_loss(W)) 
        
            logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")

        return logits

    # def bi-LSTM forward network
    def bilstm_forward(self, x, is_training=True, update_batch_stats=False, add_weight_decay=False):
        # Note that this random state is used to keep the dropout out settings the same in AT and VAT training.
        dropout_rng = np.random.RandomState(1234)
        # x shape [batch_size, sequence_length, embedding_size]
        sequence_length = x.get_shape().as_list()[1]
        
        # convert x's shape to list as time_steps * [batch_size, embedding_size]
        x = tf.unstack(x, sequence_length, 1)

        # the basic lstm cell for forward and backward respectively
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_units, reuse=self.reuse_lstm, forget_bias=0.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_units, reuse=self.reuse_lstm, forget_bias=0.0)

        # the bidirectional lstm
        lstm_out, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, 
                                                                               lstm_bw_cell, 
                                                                               x,
                                                                               dtype=tf.float32)
        
        # The first time step of bw output.
        first_bw_output = tf.split(lstm_out[0], 2, axis=1)[1]
    
        # The last time step of fw output.
        last_fw_output = tf.split(lstm_out[sequence_length-1], 2, axis=1)[0]
    
        # Concate the two lstm outputs.
        output = tf.concat([first_bw_output, last_fw_output], axis=1)

        # apply dropout to the output
        output = tf.nn.dropout(output, self.dropout_keep_prob, seed=dropout_rng.randint(123456))
        
        softmax_input = output
        input_shape = softmax_input.get_shape().as_list()[1]
        
        # Hidden layer
        layer_weights = tf.get_variable('softmax_hidden_w', 
                                        [input_shape, self.softmax_hidden_units],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=self.rng.randint(123456)))
                                                         
        layer_biases = tf.get_variable('softmax_hidden_b', 
                                       [self.softmax_hidden_units], 
                                       initializer=tf.zeros_initializer())

        # fully collected layer
        layer = tf.nn.xw_plus_b(softmax_input, layer_weights, layer_biases)
        layer = tf.nn.relu(layer)
        
        layer = tf.nn.dropout(layer, self.dropout_keep_prob, seed=dropout_rng.randint(123456))
    
        # Output layer
        output_weights = tf.get_variable('softmax_output_w', 
                                         [self.softmax_hidden_units, self.params.num_classes],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=self.rng.randint(123456)))
        output_biases = tf.get_variable('softmax_output_b', [self.params.num_classes], 
                                        initializer=tf.zeros_initializer())
        # apply weight decay to the two fully collected layers
        if add_weight_decay:
            tf.add_to_collection('weight_decays', tf.nn.l2_loss(layer_weights)) 
            tf.add_to_collection('weight_decays', tf.nn.l2_loss(output_weights))
            
        with tf.variable_scope("bilstm-output"):
            logits = tf.nn.xw_plus_b(layer, output_weights, output_biases, name="logits")
    
        # Output is raw logits without softmax
        return logits
        
        
        
        
        

vat_distributed_train
import os
import sys
import numpy as np
import random
import time
from datetime import datetime
import datahelper.datloader as loader
import pickle as cPickle
from model import DeepModel
from datahelper.dataset_utils import inputs, get_tfrecord_sample_counts, get_tfrecord_classes_counts
import tensorflow as tf


class VAT(object):
    """
    Network architecture.
    """
    def __init__(self, params = None):
        if params==None:
            return

        self.params = params
        if not os.path.isdir(self.params.model_path):
            os.makedirs(self.params.model_path)

        # load configure file
        self.config = loader.load_config(params.config)
        # load vocabulary file
        self.vocabulary = loader.json_load_vocab(self.config['resource_folder']+'/'+self.config['d2c_dict_fname'])
        self.params.vocab_len = len(self.vocabulary)
        self.params.doc_maxlen = self.config['doc_maxlen']

        self.params.model_restore_path = str(self.params.model_restore_path)
        # pre-trained word embedding
        self.params.pretrain_word2vec_path = str(self.params.pretrain_word2vec_path)
        self.params.lr_decay_method = str(self.params.lr_decay_method)

        # output the parameters
        print('=======================================================\n')
        print('Parameters\n')
        print('Model path: '+self.params.model_path)
        print('Model restore path: '+self.params.model_restore_path)
        print('Config name: '+self.params.config)
        print('Model name: '+self.params.model_method)
        print('Cost type: '+self.params.cost_type)
        print('Log dir: '+self.params.log_dir)
        print('Pretrained word2vec path: '+str(self.params.pretrain_word2vec_path))
        print('Learning rate decay method: '+str(self.params.lr_decay_method))
        print('Train epoch:'+str(self.params.num_epochs))
        print('Batch size:'+str(self.params.batch_size))
        print('Unlabled batch size:'+str(self.params.ul_batch_size))
        print('Test batch size:'+str(self.params.test_batch_size))
        print('Embedding dim: '+str(self.params.embedding_size))
        print('LSTM hidden dim: '+str(self.params.lstm_hidden_units))
        print('Softmax hidden dim: '+str(self.params.softmax_hidden_units))
        print('Dropout keep prob: '+str(self.params.dropout_keep_prob))
        print('Wrod dropout keep prob: '+str(self.params.word_dropout_keep_prob))
        print('Initial weights seed: '+str(self.params.init_weights_seed))
        print('Vocab len: '+str(self.params.vocab_len))
        print('Doc max len: '+str(self.params.doc_maxlen))
        print('AT/VAT epsilon: '+str(self.params.vat_epsilon))
        print('AT/VAT lambda: '+str(self.params.vat_lambda))
        print('L2 reg lambda: '+str(self.params.l2_reg_lambda))
        print('Early stop patience: '+str(self.params.early_stop_patience))
        print('Start learning rate: '+str(self.params.start_learning_rate))
        print('Log device placement: '+str(self.params.log_device_placement))
        print('Optimizer: '+self.params.optimizer)
        print('=======================================================\n')

        # set pretrained word vector
        self.init_w2v_weight = None
        if not(self.params.pretrain_word2vec_path.lower() == 'none') and not(self.model_exist()):
            print("\nLoading pretrained word vector...")
            self.init_w2v_weight = loader.load_pretrain_word2vec(self.params.pretrain_word2vec_path, self.vocabulary)

            # set params.embedding_size to the pre-trained embedding size
            if not(self.params.embedding_size == self.init_w2v_weight.shape[1]):
                print("Convert embedding dimension %d -> %d"%(self.params.embedding_size, self.init_w2v_weight.shape[1]))
                self.params.embedding_size = self.init_w2v_weight.shape[1]

            print("")
        
        # set learning rate
        default_lr = {'rmsprop': 0.001, 'adadelta': 0.01, 'adam': 0.0001}

        # set the optimizer
        self.optimizer_name = self.params.optimizer
        if self.params.optimizer not in default_lr.keys():
            self.optimizer_name = 'adam'

        # set the start_learning_rate
        self.start_learning_rate = self.params.start_learning_rate
        if self.params.start_learning_rate is None:
            self.start_learning_rate = default_lr[self.optimizer_name]

        # set optimizer
        if self.params.lr_decay_method.lower() == 'none':
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer(learning_rate=self.start_learning_rate, decay=0.9, epsilon=1e-6),
                      'adadelta': tf.train.AdadeltaOptimizer(learning_rate=self.start_learning_rate, rho=0.95, epsilon=1e-8),
                      'adam': tf.train.AdamOptimizer(learning_rate=self.start_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                     }
        else:
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer,
                      'adadelta': tf.train.AdadeltaOptimizer,
                      'adam': tf.train.AdamOptimizer
                     }

        self.optimizer = optims[self.optimizer_name]

    def train(self, target, cluster_spec):
        # Number of workers and parameter servers are inferred from the workers and ps
        # hosts string.
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        
        # Choose worker 0 as the chief. Note that any worker could be the chief
        # but there should be only one chief.
        is_chief = (self.params.task_id == 0)

        # get the number of classes
        self.params.num_classes = get_tfrecord_classes_counts(self.config['tf_train_data']) 
        with tf.Graph().as_default() as g:
        
            with tf.variable_scope("Input") as scope:
                # define input
                self.input_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_x")
                self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name="input_y")
                self.input_ul_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_ul_x")

            print("\nCreating iterated batch tensor node...")
            with tf.device("/cpu:0"):
                # train data size, test data size, validate data size
                num_train_size = get_tfrecord_sample_counts(self.config['tf_train_data'])
                num_test_size  = get_tfrecord_sample_counts(self.config['tf_test_data'])
                num_val_size   = get_tfrecord_sample_counts(self.config['tf_val_data'])

                # a batch of training data, test data or validate data
                train_x, train_y = inputs(batch_size=self.params.batch_size,
                                          filename=self.config['tf_train_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=self.params.num_classes,
                                          shuffle=True)
                if is_chief:
                    test_x, test_y   = inputs(batch_size=self.params.test_batch_size,
                                              filename=self.config['tf_test_data'],
                                              DOC_LEN=self.params.doc_maxlen,
                                              NUM_CLASSES=self.params.num_classes,
                                              shuffle=True)
                    val_x, val_y     = inputs(batch_size=self.params.batch_size,
                                              filename=self.config['tf_val_data'],
                                              DOC_LEN=self.params.doc_maxlen,
                                              NUM_CLASSES=self.params.num_classes,
                                              shuffle=True)
                
                unsup_x = None
                # in vat, vatent setting, unsupervised data is needed
                if self.params.cost_type.lower() in ['vat', 'vatent']: 
                    num_unsup_size   = get_tfrecord_sample_counts(self.config['tf_unsup_data'])
                    # input unsupervised data
                    unsup_x      = inputs(batch_size=self.params.ul_batch_size,
                                          filename=self.config['tf_unsup_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=-1,  #set <0 for unsup data
                                          shuffle=True)

            # output the data size
            print("")
            print("Training data size: %d"%(num_train_size))
            print("Testing data size: %d"%(num_test_size))
            print("Validation data size: %d"%(num_val_size))
            if self.params.cost_type.lower() in ['vat', 'vatent']: 
                print("Unsupervised data size: %d"%(num_unsup_size))

            print("\nBuilding tensorflow deep model...")
            # set the device
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.params.task_id, cluster=cluster_spec)):
                with tf.variable_scope("DeepModel"):
                # Build training graph
                    deepmodel_train = DeepModel(train_x, train_y, unsup_x, self.params, 
                                                dropout_keep_prob = self.params.dropout_keep_prob,
                                                word_dropout_keep_prob = self.params.word_dropout_keep_prob,
                                                is_training=True)

                    if is_chief:
                        # reuse variables
                        scope = tf.get_variable_scope()
                        scope.reuse_variables()
                        # Build VAT model
                        self.vat_model = DeepModel(self.input_x, self.input_y, self.input_ul_x, self.params,
                                                   init_w2v = self.init_w2v_weight, dropout_keep_prob = 1.,
                                                   word_dropout_keep_prob = 1., reuse_lstm=True)
                    
                        # Build eval graph
                        deepmodel_val  = DeepModel(val_x, val_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., reuse_lstm=True)
                        deepmodel_test = DeepModel(test_x, test_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., reuse_lstm=True)

                # build global_step variable
                global_step = tf.get_variable(
                              name="global_step",
                              shape=[],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0),
                              trainable=False)
                
                if self.params.lr_decay_method.lower() == 'none':
                    # constant learning rate
                    learning_rate = tf.constant(self.start_learning_rate)
                    _optimizer = self.optimizer
                elif self.params.lr_decay_method == 'exp':
                    num_iter_per_epoch = num_train_size/self.params.batch_size
                    # exponential_decay learning rate
                    learning_rate = tf.train.exponential_decay(self.start_learning_rate,
                                                               global_step,
                                                               decay_steps = num_iter_per_epoch/10,
                                                               decay_rate = 0.998,
                                                               staircase=True)
                    _optimizer = self.optimizer(learning_rate)

                # define the Synchronous optimizer
                opt = tf.train.SyncReplicasOptimizer(_optimizer,
                                                     replicas_to_aggregate=self.params.replicas_to_aggregate,
                                                     total_num_replicas=num_workers)

                # compute gradients
                grads_and_vars = opt.compute_gradients(deepmodel_train.loss, tf.trainable_variables())

                # define the training operation
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

                # Get chief queue_runners and init_tokens, which is used to synchronize
                # replicas. More details can be found in SyncReplicasOptimizer.
                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
                
                init_op = tf.global_variables_initializer()
            
                # tf saver
                self.tf_saver = tf.train.Saver(tf.global_variables())
                # define a supervisor to control the distributed training
                sv = tf.train.Supervisor(is_chief=is_chief,
                                     init_op=init_op,
                                     global_step=global_step,
                                     logdir=None,
                                     summary_op=None,
                                     saver=None,
                                     save_model_secs=0,
                                     save_summaries_secs=0)
                print('%s Supervisor' % datetime.now())
                print("\nTraining with loss: " + self.params.cost_type.upper() + "...")
            
                sess_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)

                # create a seesion
                sess = sv.prepare_or_wait_for_session(target, config=sess_config)
                self.tf_session = sess
                if is_chief:
                    sv.start_queue_runners(sess, chief_queue_runners)
                    sess.run(init_tokens_op)

                # restore model
                if self.model_exist():
                    self.load_model()
                    print("\nModel is restored from "+self.params.model_restore_path)
                
                if is_chief:
                    # define output graph name 
                    self.output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                        sess.graph_def, 
                                        output_node_names=['DeepModel/output/scores'])

                # the batch num in one epoch
                num_iter_per_epoch = num_train_size/self.params.batch_size
                early_stop_count = 0
                min_loss = np.Inf
                train_log_freq = num_iter_per_epoch/10
                test_epoch_freq = 5

                # run for num_epochs
                for epoch in range(1, self.params.num_epochs+1):
                    train_loss = 0
                    start_time = time.time()
                    print("\n==================== Epoch %d =====================\n"%epoch)

                    for batch_idx in range(num_iter_per_epoch):
                        # run training
                        _, batch_loss, batch_acc, lr, gs = sess.run([train_op, 
                                                                     deepmodel_train.loss,
                                                                     deepmodel_train.accuracy, 
                                                                     learning_rate, global_step])
                        train_loss += batch_loss

                        # log for training
                        if batch_idx % train_log_freq == train_log_freq - 1:
                            print("global_step: %6d  batch_loss: %6.3f  batch_acc: %6.2f%%  lr: %7.6f" % \
                                  (gs, batch_loss, batch_acc*100, lr))
                    duration = time.time() - start_time
                    print("duration: %.2f secs" % duration)
                    train_loss /= num_iter_per_epoch               
                    print("\n"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    
                    if is_chief:
                        # evaluate the model
                        val_accuracy, val_loss = self.evaluate(sess, deepmodel_val,
                                                        num_val_size/self.params.batch_size)
                        print("Epoch: %4d, train_loss: %6.3f, val_loss: %6.3f, val_acc: %5.2f%%" % \
                         (epoch, train_loss, val_loss, val_accuracy*100))


                        if epoch % test_epoch_freq == 0:
                            # evaluate the model on test dataset
                            accuracy, loss = self.evaluate(sess, deepmodel_test,
                                                       num_test_size/self.params.test_batch_size)
                            print("\n* Test epoch %3d, test_loss: %6.3f, test_acc: %5.2f%%" % (epoch, loss,
                                                                                       accuracy*100))
                        # if get a small val_loss, save the model and reset early_stop_count to 0
                        if val_loss < min_loss:
                            min_loss = val_loss
                            self.save_model("best")
                            print("New best model is saved!")

                            early_stop_count = 0
                        else:
                            early_stop_count += 1

                        # the early stopped condition of training
                        if early_stop_count > self.params.early_stop_patience:
                            print("Training is early stopped!")
                            break

                print("\n===================================================")
                print("\nTraining is done!")
                if is_chief:
                    self.save_model("last")
                
                    print("\nRestore best model on evaluation set.")
                    ckpt_model_name = self.params.model_path + '/best-vat_model.ckpt'
                    self.load_model(ckpt_model_name=ckpt_model_name)
            
                    print("\nTesting on best model...")
                    accuracy, loss = self.evaluate(sess, deepmodel_test, num_test_size/self.params.test_batch_size)
                    print("\n* Test on best model, test_loss: %6.3f, test_acc: %5.2f%%" % (loss, accuracy*100))

                    self.save_model()
                    print("\nTesting is done!")
                
                sess.close()

    def evaluate(self, sess, deepmodel, nbatch):
        """
        Evaluate the model
        """
        correct_list = []
        loss_sum = 0
        # evaluate on nbatch batches data
        for batch_idx in range(nbatch):
            loss, correct = sess.run([deepmodel.loss, deepmodel.correct_predictions])
            correct_list += correct.tolist()
            loss_sum += loss

        ndata = len(correct_list)
        accuracy = float(sum(correct_list)) / ndata
        # the average loss on each batch
        loss = loss_sum / nbatch
        return accuracy, loss

    def save_params(self):
        """
        Dump model parameters via cPickle
        """
        with open(self.params.model_path + '/vat_params.pkl', 'wb') as f:
            cPickle.dump(self.params, f)

    def save_model(self, aux_name=""):
        """
        save model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"

        pb_model_name = self.params.model_path + '/' + aux_name + 'vat_model.pb'
        with tf.gfile.FastGFile(pb_model_name, mode='wb') as f:
            f.write(self.output_graph_def.SerializeToString())

        ckpt_model_name = self.params.model_path + '/' + aux_name + 'vat_model.ckpt'
        self.tf_saver.save(self.tf_session, ckpt_model_name)

    def load_model(self, aux_name="", ckpt_model_name = None):
        """
        load model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'
        
        self.tf_saver.restore(self.tf_session, ckpt_model_name)

    def model_exist(self, aux_name="", ckpt_model_name = None):
        """
            Test whether model exists
        """
        if self.params.model_restore_path.lower() == 'none':
            return False

        if not(aux_name == ""):
            aux_name = aux_name+"-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'

        if os.path.exists(ckpt_model_name+'.meta') and os.path.exists(ckpt_model_name+'.index'):
            return True
        else:
            return False


vat_distributed_train2
import os
import sys
import numpy as np
import random
import time
from datetime import datetime
import datahelper.datloader as loader
import pickle as cPickle
from model import DeepModel
from datahelper.dataset_utils import inputs, get_tfrecord_sample_counts, get_tfrecord_classes_counts
import tensorflow as tf


class VAT(object):
    """
    Network architecture.
    """
    def __init__(self, params = None):
        if params==None:
            return

        self.params = params
        if not os.path.isdir(self.params.model_path):
            os.makedirs(self.params.model_path)

        # load configure file
        self.config = loader.load_config(params.config)
        # load vocabulary file
        self.vocabulary = loader.json_load_vocab(self.config['resource_folder']+'/'+self.config['d2c_dict_fname'])
        self.params.vocab_len = len(self.vocabulary)
        self.params.doc_maxlen = self.config['doc_maxlen']

        self.params.model_restore_path = str(self.params.model_restore_path)
        # pre-trained word embedding
        self.params.pretrain_word2vec_path = str(self.params.pretrain_word2vec_path)
        self.params.lr_decay_method = str(self.params.lr_decay_method)

        # output the parameters
        print('=======================================================\n')
        print('Parameters\n')
        print('Model path: '+self.params.model_path)
        print('Model restore path: '+self.params.model_restore_path)
        print('Config name: '+self.params.config)
        print('Model name: '+self.params.model_method)
        print('Cost type: '+self.params.cost_type)
        print('Log dir: '+self.params.log_dir)
        print('Pretrained word2vec path: '+str(self.params.pretrain_word2vec_path))
        print('Learning rate decay method: '+str(self.params.lr_decay_method))
        print('Train epoch:'+str(self.params.num_epochs))
        print('Batch size:'+str(self.params.batch_size))
        print('Unlabled batch size:'+str(self.params.ul_batch_size))
        print('Test batch size:'+str(self.params.test_batch_size))
        print('Embedding dim: '+str(self.params.embedding_size))
        print('LSTM hidden dim: '+str(self.params.lstm_hidden_units))
        print('Softmax hidden dim: '+str(self.params.softmax_hidden_units))
        print('Dropout keep prob: '+str(self.params.dropout_keep_prob))
        print('Wrod dropout keep prob: '+str(self.params.word_dropout_keep_prob))
        print('Initial weights seed: '+str(self.params.init_weights_seed))
        print('Vocab len: '+str(self.params.vocab_len))
        print('Doc max len: '+str(self.params.doc_maxlen))
        print('AT/VAT epsilon: '+str(self.params.vat_epsilon))
        print('AT/VAT lambda: '+str(self.params.vat_lambda))
        print('L2 reg lambda: '+str(self.params.l2_reg_lambda))
        print('Early stop patience: '+str(self.params.early_stop_patience))
        print('Start learning rate: '+str(self.params.start_learning_rate))
        print('Log device placement: '+str(self.params.log_device_placement))
        print('Optimizer: '+self.params.optimizer)
        print('=======================================================\n')

        # set pretrained word vector
        self.init_w2v_weight = None
        if not(self.params.pretrain_word2vec_path.lower() == 'none') and not(self.model_exist()):
            print("\nLoading pretrained word vector...")
            self.init_w2v_weight = loader.load_pretrain_word2vec(self.params.pretrain_word2vec_path, self.vocabulary)

            # set params.embedding_size to the pre-trained embedding size
            if not(self.params.embedding_size == self.init_w2v_weight.shape[1]):
                print("Convert embedding dimension %d -> %d"%(self.params.embedding_size, self.init_w2v_weight.shape[1]))
                self.params.embedding_size = self.init_w2v_weight.shape[1]

            print("")
        
        # set learning rate
        default_lr = {'rmsprop': 0.001, 'adadelta': 0.01, 'adam': 0.0001}

        # set the optimizer
        self.optimizer_name = self.params.optimizer
        if self.params.optimizer not in default_lr.keys():
            self.optimizer_name = 'adam'

        # set the start_learning_rate
        self.start_learning_rate = self.params.start_learning_rate
        if self.params.start_learning_rate is None:
            self.start_learning_rate = default_lr[self.optimizer_name]

        # set optimizer
        if self.params.lr_decay_method.lower() == 'none':
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer(learning_rate=self.start_learning_rate, decay=0.9, epsilon=1e-6),
                      'adadelta': tf.train.AdadeltaOptimizer(learning_rate=self.start_learning_rate, rho=0.95, epsilon=1e-8),
                      'adam': tf.train.AdamOptimizer(learning_rate=self.start_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                     }
        else:
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer,
                      'adadelta': tf.train.AdadeltaOptimizer,
                      'adam': tf.train.AdamOptimizer
                     }

        self.optimizer = optims[self.optimizer_name]

    def train(self, target, cluster_spec):
        # Number of workers and parameter servers are inferred from the workers and ps
        # hosts string.
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        
        # Choose worker 0 as the chief. Note that any worker could be the chief
        # but there should be only one chief.
        is_chief = (self.params.task_id == 0)

        # get the number of classes
        self.params.num_classes = get_tfrecord_classes_counts(self.config['tf_train_data']) 
        with tf.Graph().as_default() as g:
        
            with tf.variable_scope("Input") as scope:
                # define input
                self.input_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_x")
                self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name="input_y")
                self.input_ul_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_ul_x")

            print("\nCreating iterated batch tensor node...")
            with tf.device("/cpu:0"):
                # train data size, test data size, validate data size
                num_train_size = get_tfrecord_sample_counts(self.config['tf_train_data'])
                num_test_size  = get_tfrecord_sample_counts(self.config['tf_test_data'])
                num_val_size   = get_tfrecord_sample_counts(self.config['tf_val_data'])

                # a batch of training data, test data or validate data
                train_x, train_y = inputs(batch_size=self.params.batch_size,
                                          filename=self.config['tf_train_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=self.params.num_classes,
                                          shuffle=True)
                if is_chief:
                    test_x, test_y   = inputs(batch_size=self.params.test_batch_size,
                                              filename=self.config['tf_test_data'],
                                              DOC_LEN=self.params.doc_maxlen,
                                              NUM_CLASSES=self.params.num_classes,
                                              shuffle=True)
                    val_x, val_y     = inputs(batch_size=self.params.batch_size,
                                              filename=self.config['tf_val_data'],
                                              DOC_LEN=self.params.doc_maxlen,
                                              NUM_CLASSES=self.params.num_classes,
                                              shuffle=True)
                
                unsup_x = None
                # in vat, vatent setting, unsupervised data is needed
                if self.params.cost_type.lower() in ['vat', 'vatent']: 
                    num_unsup_size   = get_tfrecord_sample_counts(self.config['tf_unsup_data'])
                    # input unsupervised data
                    unsup_x      = inputs(batch_size=self.params.ul_batch_size,
                                          filename=self.config['tf_unsup_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=-1,  #set <0 for unsup data
                                          shuffle=True)

            # output the data size
            print("")
            print("Training data size: %d"%(num_train_size))
            print("Testing data size: %d"%(num_test_size))
            print("Validation data size: %d"%(num_val_size))
            if self.params.cost_type.lower() in ['vat', 'vatent']: 
                print("Unsupervised data size: %d"%(num_unsup_size))

            print("\nBuilding tensorflow deep model...")
            # set the device
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.params.task_id, cluster=cluster_spec)):
                with tf.variable_scope("DeepModel"):
                # Build training graph
                    deepmodel_train = DeepModel(train_x, train_y, unsup_x, self.params, 
                                                dropout_keep_prob = self.params.dropout_keep_prob,
                                                word_dropout_keep_prob = self.params.word_dropout_keep_prob,
                                                is_training=True)

                    if is_chief:
                        # reuse variables
                        scope = tf.get_variable_scope()
                        scope.reuse_variables()
                        # Build VAT model
                        self.vat_model = DeepModel(self.input_x, self.input_y, self.input_ul_x, self.params,
                                                   init_w2v = self.init_w2v_weight, dropout_keep_prob = 1.,
                                                   word_dropout_keep_prob = 1., reuse_lstm=True)
                    
                        # Build eval graph
                        deepmodel_val  = DeepModel(val_x, val_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., reuse_lstm=True)
                        deepmodel_test = DeepModel(test_x, test_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., reuse_lstm=True)

                # build global_step variable
                global_step = tf.get_variable(
                              name="global_step",
                              shape=[],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0),
                              trainable=False)
                
                if self.params.lr_decay_method.lower() == 'none':
                    # constant learning rate
                    learning_rate = tf.constant(self.start_learning_rate)
                    _optimizer = self.optimizer
                elif self.params.lr_decay_method == 'exp':
                    num_iter_per_epoch = num_train_size/self.params.batch_size
                    # exponential_decay learning rate
                    learning_rate = tf.train.exponential_decay(self.start_learning_rate,
                                                               global_step,
                                                               decay_steps = num_iter_per_epoch/10,
                                                               decay_rate = 0.998,
                                                               staircase=True)
                    _optimizer = self.optimizer(learning_rate)

                # define the Synchronous optimizer
                opt = tf.train.SyncReplicasOptimizer(_optimizer,
                                                     replicas_to_aggregate=self.params.replicas_to_aggregate,
                                                     total_num_replicas=num_workers)

                # compute gradients
                grads_and_vars = opt.compute_gradients(deepmodel_train.loss, tf.trainable_variables())

                # define the training operation
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

                # Get chief queue_runners and init_tokens, which is used to synchronize
                # replicas. More details can be found in SyncReplicasOptimizer.
                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
                
                init_op = tf.global_variables_initializer()
            
                # tf saver
                self.tf_saver = tf.train.Saver(tf.global_variables())
                # define a supervisor to control the distributed training
                sv = tf.train.Supervisor(is_chief=is_chief,
                                     init_op=init_op,
                                     global_step=global_step,
                                     logdir=None,
                                     summary_op=None,
                                     saver=None,
                                     save_model_secs=0,
                                     save_summaries_secs=0)
                print('%s Supervisor' % datetime.now())
                print("\nTraining with loss: " + self.params.cost_type.upper() + "...")
            
                sess_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)

                # create a seesion
                sess = sv.prepare_or_wait_for_session(target, config=sess_config)
                self.tf_session = sess
                if is_chief:
                    sv.start_queue_runners(sess, chief_queue_runners)
                    sess.run(init_tokens_op)

                # restore model
                if self.model_exist():
                    self.load_model()
                    print("\nModel is restored from "+self.params.model_restore_path)
                
                if is_chief:
                    # define output graph name 
                    self.output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                        sess.graph_def, 
                                        output_node_names=['DeepModel/output/scores'])

                # the batch num in one epoch
                num_iter_per_epoch = num_train_size/self.params.batch_size
                early_stop_count = 0
                min_loss = np.Inf
                train_log_freq = num_iter_per_epoch/10
                test_epoch_freq = 5

                # run for num_epochs
                for epoch in range(1, self.params.num_epochs+1):
                    train_loss = 0
                    start_time = time.time()
                    print("\n==================== Epoch %d =====================\n"%epoch)

                    for batch_idx in range(num_iter_per_epoch):
                        # run training
                        _, batch_loss, batch_acc, lr, gs = sess.run([train_op, 
                                                                     deepmodel_train.loss,
                                                                     deepmodel_train.accuracy, 
                                                                     learning_rate, global_step])
                        train_loss += batch_loss

                        # log for training
                        if batch_idx % train_log_freq == train_log_freq - 1:
                            print("global_step: %6d  batch_loss: %6.3f  batch_acc: %6.2f%%  lr: %7.6f" % \
                                  (gs, batch_loss, batch_acc*100, lr))
                    duration = time.time() - start_time
                    print("duration: %.2f secs" % duration)
                    train_loss /= num_iter_per_epoch               
                    print("\n"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    
                    if is_chief:
                        # evaluate the model
                        val_accuracy, val_loss = self.evaluate(sess, deepmodel_val,
                                                        num_val_size/self.params.batch_size)
                        print("Epoch: %4d, train_loss: %6.3f, val_loss: %6.3f, val_acc: %5.2f%%" % \
                         (epoch, train_loss, val_loss, val_accuracy*100))


                        if epoch % test_epoch_freq == 0:
                            # evaluate the model on test dataset
                            accuracy, loss = self.evaluate(sess, deepmodel_test,
                                                       num_test_size/self.params.test_batch_size)
                            print("\n* Test epoch %3d, test_loss: %6.3f, test_acc: %5.2f%%" % (epoch, loss,
                                                                                       accuracy*100))
                        # if get a small val_loss, save the model and reset early_stop_count to 0
                        if val_loss < min_loss:
                            min_loss = val_loss
                            self.save_model("best")
                            print("New best model is saved!")

                            early_stop_count = 0
                        else:
                            early_stop_count += 1

                        # the early stopped condition of training
                        if early_stop_count > self.params.early_stop_patience:
                            print("Training is early stopped!")
                            break

                print("\n===================================================")
                print("\nTraining is done!")
                if is_chief:
                    self.save_model("last")
                
                    print("\nRestore best model on evaluation set.")
                    ckpt_model_name = self.params.model_path + '/best-vat_model.ckpt'
                    self.load_model(ckpt_model_name=ckpt_model_name)
            
                    print("\nTesting on best model...")
                    accuracy, loss = self.evaluate(sess, deepmodel_test, num_test_size/self.params.test_batch_size)
                    print("\n* Test on best model, test_loss: %6.3f, test_acc: %5.2f%%" % (loss, accuracy*100))

                    self.save_model()
                    print("\nTesting is done!")
                
                sess.close()

    def evaluate(self, sess, deepmodel, nbatch):
        """
        Evaluate the model
        """
        correct_list = []
        loss_sum = 0
        # evaluate on nbatch batches data
        for batch_idx in range(nbatch):
            loss, correct = sess.run([deepmodel.loss, deepmodel.correct_predictions])
            correct_list += correct.tolist()
            loss_sum += loss

        ndata = len(correct_list)
        accuracy = float(sum(correct_list)) / ndata
        # the average loss on each batch
        loss = loss_sum / nbatch
        return accuracy, loss

    def save_params(self):
        """
        Dump model parameters via cPickle
        """
        with open(self.params.model_path + '/vat_params.pkl', 'wb') as f:
            cPickle.dump(self.params, f)

    def save_model(self, aux_name=""):
        """
        save model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"

        pb_model_name = self.params.model_path + '/' + aux_name + 'vat_model.pb'
        with tf.gfile.FastGFile(pb_model_name, mode='wb') as f:
            f.write(self.output_graph_def.SerializeToString())

        ckpt_model_name = self.params.model_path + '/' + aux_name + 'vat_model.ckpt'
        self.tf_saver.save(self.tf_session, ckpt_model_name)

    def load_model(self, aux_name="", ckpt_model_name = None):
        """
        load model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'
        
        self.tf_saver.restore(self.tf_session, ckpt_model_name)

    def model_exist(self, aux_name="", ckpt_model_name = None):
        """
            Test whether model exists
        """
        if self.params.model_restore_path.lower() == 'none':
            return False

        if not(aux_name == ""):
            aux_name = aux_name+"-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'

        if os.path.exists(ckpt_model_name+'.meta') and os.path.exists(ckpt_model_name+'.index'):
            return True
        else:
            return False

vat_train
import os
import sys
import numpy as np
import random
import time
import datahelper.datloader as loader
import pickle as cPickle
from model import DeepModel
from datahelper.dataset_utils import inputs, get_tfrecord_sample_counts, get_tfrecord_classes_counts
import tensorflow as tf


class VAT(object):
    """
    Network architecture.
    """
    def __init__(self, params = None):
        if params==None:
            return

        self.params = params
        if not os.path.isdir(self.params.model_path):
            os.makedirs(self.params.model_path)

        # load configure file
        self.config = loader.load_config(params.config)
        # load vocabulary file
        self.vocabulary = loader.json_load_vocab(self.config['resource_folder']+'/'+self.config['d2c_dict_fname'])
        # the size of vocabulary
        self.params.vocab_len = len(self.vocabulary)
        # the maximun length of document
        self.params.doc_maxlen = self.config['doc_maxlen']

        self.params.model_restore_path = str(self.params.model_restore_path)
        # the path of pre-trained word vector
        self.params.pretrain_word2vec_path = str(self.params.pretrain_word2vec_path)
        # learning rate decay method
        self.params.lr_decay_method = str(self.params.lr_decay_method)

        # output all the parameters
        print('=======================================================\n')
        print('Parameters\n')
        print('Device: '+self.params.device)
        print('Model path: '+self.params.model_path)
        print('Model restore path: '+self.params.model_restore_path)
        print('Config name: '+self.params.config)
        print('Model name: '+self.params.model_method)
        print('Cost type: '+self.params.cost_type)
        print('Log dir: '+self.params.log_dir)
        print('Pretrained word2vec path: '+self.params.pretrain_word2vec_path)
        print('Learning rate decay method: '+self.params.lr_decay_method)
        print('Train epoch:'+str(self.params.num_epochs))
        print('Batch size:'+str(self.params.batch_size))
        print('Unlabled batch size:'+str(self.params.ul_batch_size))
        print('Test batch size:'+str(self.params.test_batch_size))
        print('Embedding dim: '+str(self.params.embedding_size))
        print('LSTM hidden dim: '+str(self.params.lstm_hidden_units))
        print('Softmax hidden dim: '+str(self.params.softmax_hidden_units))
        print('Dropout keep prob: '+str(self.params.dropout_keep_prob))
        print('Wrod dropout keep prob: '+str(self.params.word_dropout_keep_prob))
        print('Initial weights seed: '+str(self.params.init_weights_seed))
        print('Vocab len: '+str(self.params.vocab_len))
        print('Doc max len: '+str(self.params.doc_maxlen))
        print('AT/VAT epsilon: '+str(self.params.vat_epsilon))
        print('AT/VAT lambda: '+str(self.params.vat_lambda))
        print('L2 reg lambda: '+str(self.params.l2_reg_lambda))
        print('Early stop patience: '+str(self.params.early_stop_patience))
        print('Start learning rate: '+str(self.params.start_learning_rate))
        print('Log device placement: '+str(self.params.log_device_placement))
        print('Optimizer: '+self.params.optimizer)
        print('=======================================================\n')

        # set random seed
        self.rng = np.random.RandomState(self.params.init_weights_seed)

        # set pretrained word vector
        self.init_w2v_weight = None
        if not(self.params.pretrain_word2vec_path.lower() == 'none') and not(self.model_exist()):
            print("\nLoading pretrained word vector...")
            self.init_w2v_weight = loader.load_pretrain_word2vec(self.params.pretrain_word2vec_path, self.vocabulary)

            # if params.embedding_size is not equal to the pre-trained embedding size,
            # set params.embedding_size to the pre-trained embedding size
            if not(self.params.embedding_size == self.init_w2v_weight.shape[1]):
                print("Convert embedding dimension %d -> %d"%(self.params.embedding_size, self.init_w2v_weight.shape[1]))
                self.params.embedding_size = self.init_w2v_weight.shape[1]

            print("")

        # set learning rate
        default_lr = {'rmsprop': 0.001, 'adadelta': 0.01, 'adam': 0.0001}

        # set the optimizer
        self.optimizer_name = self.params.optimizer
        if self.params.optimizer not in default_lr.keys():
            self.optimizer_name = 'adam'

        # set the start_learning_rate
        self.start_learning_rate = self.params.start_learning_rate
        if self.params.start_learning_rate is None:
            self.start_learning_rate = default_lr[self.optimizer_name]

        # set optimizer
        if self.params.lr_decay_method.lower() == 'none':
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer(learning_rate=self.start_learning_rate, decay=0.9, epsilon=1e-6),
                      'adadelta': tf.train.AdadeltaOptimizer(learning_rate=self.start_learning_rate, rho=0.95, epsilon=1e-8),
                      'adam': tf.train.AdamOptimizer(learning_rate=self.start_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
                     }
        else:
            optims = {
                      'rmsprop': tf.train.RMSPropOptimizer,
                      'adadelta': tf.train.AdadeltaOptimizer,
                      'adam': tf.train.AdamOptimizer
                     }

        self.optimizer = optims[self.optimizer_name]
        

    def train(self):
        # get the number of classes
        self.params.num_classes = get_tfrecord_classes_counts(self.config['tf_train_data']) 
        print("\n train_num_classes %3d" % self.params.num_classes)  # maoxue
        with tf.Graph().as_default() as g:
            with tf.variable_scope("Input") as scope:
                # define input
                self.input_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_x")
                self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name="input_y")
                self.input_ul_x = tf.placeholder(tf.int32, [None, self.params.doc_maxlen], name="input_ul_x")

            print("\nCreating iterated batch tensor node...")
            with tf.device("/cpu:0"):
                # train data size, test data size, validate data size
                num_train_size = get_tfrecord_sample_counts(self.config['tf_train_data'])
                num_test_size  = get_tfrecord_sample_counts(self.config['tf_test_data'])
                num_val_size   = get_tfrecord_sample_counts(self.config['tf_val_data'])

                # a batch of training data, test data or validate data
                train_x, train_y = inputs(batch_size=self.params.batch_size,
                                          filename=self.config['tf_train_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=self.params.num_classes,
                                          shuffle=True,
                                          seed=self.rng.randint(123456))
                test_x, test_y = inputs(batch_size=self.params.test_batch_size,
                                        filename=self.config['tf_test_data'],
                                        DOC_LEN=self.params.doc_maxlen,
                                        NUM_CLASSES=self.params.num_classes,
                                        shuffle=False)
                val_x, val_y = inputs(batch_size=self.params.batch_size,
                                      filename=self.config['tf_val_data'],
                                      DOC_LEN=self.params.doc_maxlen,
                                      NUM_CLASSES=self.params.num_classes,
                                      shuffle=False)
                
                unsup_x = None

                # in vat, vatent setting, unsupervised data is needed   
                if self.params.cost_type.lower() in ['vat', 'vatent']: 
                    num_unsup_size   = get_tfrecord_sample_counts(self.config['tf_unsup_data'])
                    # input unsupervised data
                    unsup_x      = inputs(batch_size=self.params.ul_batch_size,
                                          filename=self.config['tf_unsup_data'],
                                          DOC_LEN=self.params.doc_maxlen,
                                          NUM_CLASSES=-1,  #set <0 for unsup data
                                          shuffle=True,
                                          seed=self.rng.randint(123456))

            # output the data size
            print("")
            print("Training data size: %d"%(num_train_size))
            print("Testing data size: %d"%(num_test_size))
            print("Validation data size: %d"%(num_val_size))
            if self.params.cost_type.lower() in ['vat', 'vatent']: 
                print("Unsupervised data size: %d"%(num_unsup_size))

            print("\nBuilding tensorflow deep model...")
            with tf.device(self.params.device):
                with tf.variable_scope("DeepModel"):
                    # Build VAT model
                    self.vat_model = DeepModel(self.input_x, self.input_y, self.input_ul_x, self.params, init_w2v = self.init_w2v_weight, dropout_keep_prob = 1., word_dropout_keep_prob = 1.)

                    # reuse variables
                    scope = tf.get_variable_scope()
                    scope.reuse_variables()

                    # Build training graph
                    deepmodel_train = DeepModel(train_x, train_y, unsup_x, self.params, 
                                                dropout_keep_prob = self.params.dropout_keep_prob,
                                                word_dropout_keep_prob = self.params.word_dropout_keep_prob,
                                                is_training=True, reuse_lstm=True)

                    # Build eval graph
                    deepmodel_val  = DeepModel(val_x, val_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., is_training=False, reuse_lstm=True)
                    deepmodel_test = DeepModel(test_x, test_y, unsup_x, self.params, dropout_keep_prob=1.,
                                               word_dropout_keep_prob=1., is_training=False, reuse_lstm=True)
                # build global_step variable
                global_step = tf.get_variable(
                              name="global_step",
                              shape=[],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0),
                              trainable=False)

                if self.params.lr_decay_method.lower() == 'none':
                    # constant learning rate
                    learning_rate = tf.constant(self.start_learning_rate)
                    grads_and_vars = self.optimizer.compute_gradients(deepmodel_train.loss, tf.trainable_variables())
                    # define the training operation
                    train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                elif self.params.lr_decay_method == 'exp':
                    num_iter_per_epoch = num_train_size/self.params.batch_size
                    # exponential_decay learning rate
                    learning_rate = tf.train.exponential_decay(self.start_learning_rate,
                                                               global_step,
                                                               decay_steps = num_iter_per_epoch/10,
                                                               decay_rate = 0.998,
                                                               staircase=True)

                    # compute gradients
                    grads_and_vars = self.optimizer(learning_rate).compute_gradients(deepmodel_train.loss, tf.trainable_variables())
                    # define the training operation
                    train_op = self.optimizer(learning_rate).apply_gradients(grads_and_vars, global_step=global_step)
                else:
                    raise NotImplementedError()

                init_op = tf.global_variables_initializer()

            # tf saver
            self.tf_saver = tf.train.Saver(tf.global_variables())
            proto_config = tf.ConfigProto(
                           allow_soft_placement = True,
                           log_device_placement = self.params.log_device_placement)

            print("\nTraining with loss: " + self.params.cost_type.upper() + "...")
            # create a seesion
            with tf.Session(config = proto_config) as sess:
                # define tfrecord thread
                coord = tf.train.Coordinator()
                # start input queue runners
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)

                # run init
                sess.run(init_op)
                self.tf_session = sess
                # restore model
                if self.model_exist():
                    self.load_model()
                    print("\nModel is restored from "+self.params.model_restore_path)
                # define output graph name 
                self.output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                        sess.graph_def, 
                                        output_node_names=['DeepModel/output/scores'])
                # the batch num in one epoch
                num_iter_per_epoch = num_train_size/self.params.batch_size
                early_stop_count = 0
                min_loss = np.Inf
                train_log_freq = num_iter_per_epoch/10
                test_epoch_freq = 5

                # run for num_epochs
                for epoch in range(1, self.params.num_epochs+1):
                    train_loss = 0
                    print("\n=============================== Epoch %3d ===============================\n"%epoch)
                    for batch_idx in range(num_iter_per_epoch):
                        #jzh ***************
                        print("for "+str(batch_idx)+'-' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))   
                        # run training
                        _, batch_loss, batch_acc, lr, gs = sess.run([train_op, 
                                                                     deepmodel_train.loss, deepmodel_train.accuracy, 
                                                                     learning_rate, global_step])
                        #jzh ***************
                        print('train_acc', str(batch_loss), str(batch_acc), str(lr), str(gs))
                        train_loss += batch_loss
                        #jzh ***************
                        print('train_loss',str(train_loss))

                        # log for training
                        if batch_idx % train_log_freq == train_log_freq - 1:
                            print("global_step: %6d  batch_loss: %6.3f  batch_acc: %6.2f%%  lr: %7.6f" % \
                                  (gs, batch_loss, batch_acc*100, lr))

                    train_loss /= num_iter_per_epoch
                    # evaluate the model
                    val_accuracy, val_loss = self.evaluate(sess, deepmodel_val, num_val_size/self.params.batch_size)
                    print("\n"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    print("Epoch: %4d, train_loss: %6.3f, val_loss: %6.3f, val_acc: %6.2f%%" % \
                         (epoch, train_loss, val_loss, val_accuracy*100))

                    if epoch % test_epoch_freq == 0:
                        # evaluate the model on test dataset
                        accuracy, loss = self.evaluate(sess, deepmodel_test, num_test_size/self.params.test_batch_size)
                        print("\n* Test epoch %3d, test_loss: %6.3f, test_acc: %6.2f%%" % (epoch, loss, accuracy*100))

                    # if get a small val_loss, save the model and reset early_stop_count to 0
                    if val_loss < min_loss:
                        min_loss = val_loss
                        self.save_model("best")
                        print("New best model is saved!")
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

                    # the early stopped condition of training
                    if early_stop_count > self.params.early_stop_patience:
                        print("Training is early stopped!")
                        break

                print("\n==========================================================================")
                print("\nTraining is done!")
                self.save_model("last")
                
                print("\nRestore best model on evaluation set.")
                ckpt_model_name = self.params.model_path + '/best-vat_model.ckpt'
                self.load_model(ckpt_model_name=ckpt_model_name)

                print("\nTesting on best model...")
                t0 = time.time()#maoxue

                accuracy, loss = self.evaluate(sess, deepmodel_test, num_test_size/self.params.test_batch_size)
                print(time.time() - t0)#maoxue
                print("seconds wall time")#maoxue
                print("\n* Test on best model, test_loss: %6.3f, test_acc: %5.2f%%" % (loss, accuracy*100))
                self.save_model()
                print("\nTesting is done!")

                # close threads
                coord.request_stop()
                coord.join(threads)

    def evaluate(self, sess, deepmodel, nbatch):
        """
        Evaluate the model
        """
        correct_list = []
        loss_sum = 0

        # evaluate on nbatch batches data
        for batch_idx in range(nbatch):
            loss, correct = sess.run([deepmodel.loss, deepmodel.correct_predictions])
            correct_list += correct.tolist()
            loss_sum += loss

        ndata = len(correct_list)
        accuracy = float(sum(correct_list)) / ndata
        # the average loss on each batch
        loss = loss_sum / nbatch
        return accuracy, loss

    def save_params(self):
        """
        Dump model parameters via cPickle
        """
        with open(self.params.model_path + '/vat_params.pkl', 'wb') as f:
            cPickle.dump(self.params, f)

    def save_model(self, aux_name=""):
        """
        save model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"
        pb_model_name = self.params.model_path + '/' + aux_name + 'vat_model.pb'
        with tf.gfile.FastGFile(pb_model_name, mode='wb') as f:
            f.write(self.output_graph_def.SerializeToString())
        ckpt_model_name = self.params.model_path + '/' + aux_name + 'vat_model.ckpt'
        self.tf_saver.save(self.tf_session, ckpt_model_name)

    def load_model(self, aux_name="", ckpt_model_name = None):
        """
        load model
        """
        if not(aux_name == ""):
            aux_name = aux_name+"-"
        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'

        # restore model from ckpt_model_name
        self.tf_saver.restore(self.tf_session, ckpt_model_name)

    def model_exist(self, aux_name="", ckpt_model_name = None):
        """
        Test whether model exists
        """
        if self.params.model_restore_path.lower() == 'none':
            return False
        if not(aux_name == ""):
            aux_name = aux_name+"-"
        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'vat_model.ckpt'
        if os.path.exists(ckpt_model_name+'.meta') and os.path.exists(ckpt_model_name+'.index'):
            return True
        else:
            return False

vat_train2

import os
import sys
import numpy as np
import random
import time
import datahelper.datloader as loader
import pickle as cPickle
import json
from model import DeepModel
from datahelper.dataset_utils import inputs, get_tfrecord_sample_counts, get_tfrecord_classes_counts
import re
import tensorflow as tf
from copy import deepcopy

class VAT(object):
    """
    Network architecture.
    """

    def __init__(self, params=None):
        if params == None:
            return

        self.params = params
        self.config = loader.load_config(params.config)

        self.params.resource_path = str(self.config['resource_folder'])

        if self.params.resource_path.endswith('/'):
            self.params.resource_path = self.params.resource_path[:-1]

        self.params.active_learning_data_path = str(self.config['tf_test_data'])

        self.config['tf_train_data'] = self.params.active_learning_data_path
        self.params.model_restore_path = str(self.params.model_restore_path)

        # path of vocab and ind2cat
        self.vocabulary = loader.json_load_vocab(self.config['resource_folder']+'/'+self.config['d2c_dict_fname'])
        self.ind2cat_path = str(self.params.pretrain_word2vec_path)       #"/home/xue.mao/D2C_train_21/137int2cat.txt"
        self.params.vocab_len = len(self.vocabulary)
        self.params.doc_maxlen = self.config['doc_maxlen']
        self.params.pretrain_word2vec_path = str(self.params.pretrain_word2vec_path)


        self.params.active_learning_output_path = str(self.config['predict_save_path'])    
                                                #'/home/xue.mao/D2C_train_21/data/zx/active_learning_pred.csv'

        self.params.lr_decay_method = str(self.params.lr_decay_method)

        self.params.count_accuracy_input_tfrecords = self.params.active_learning_data_path

        #depreciate
        self.params.confusion_matrix_output_dir = '/home/xue.mao/D2C_train_21/test/confuseion_matrix_output.csv'
        self.params.output_texts_dir = '/home/xue.mao/D2C_train_21/data/zx/output.csv'


        i=1
        if i==1:
            print('=======================================================\n')
            print('Parameters\n')
            print('Device: ' + self.params.device)
            print('Resource path: ' + self.params.resource_path)
            print('Model restore path: ' + self.params.model_restore_path)
            print('Config name: ' + self.params.config)
            print('Model name: ' + self.params.model_method)
            print('Cost type: ' + self.params.cost_type)
            print('Log dir: ' + self.params.log_dir)
            print('Pretrained word2vec path: ' + self.params.pretrain_word2vec_path)
            print('Learning rate decay method: ' + self.params.lr_decay_method)
            print('LSTM cell number: ' + str(self.params.lstm_hidden_units))
            print('Train epoch:' + str(self.params.num_epochs))
            print('Batch size:' + str(self.params.batch_size))
            print('Unlabled batch size:' + str(self.params.ul_batch_size))
            print('Test batch size:' + str(self.params.test_batch_size))
            print('Embedding dim: ' + str(self.params.embedding_size))
            print('Dropout keep prob: ' + str(self.params.dropout_keep_prob))
            print('Wrod dropout keep prob: ' + str(self.params.word_dropout_keep_prob))
      
            print('Initial weights seed: ' + str(self.params.init_weights_seed))
            print('Vocab len: ' + str(self.params.vocab_len))
            print('Doc max len: ' + str(self.params.doc_maxlen))
            print('AT/VAT epsilon: ' + str(self.params.vat_epsilon))
            print('AT/VAT lambda: ' + str(self.params.vat_lambda))
            print('L2 reg lambda: ' + str(self.params.l2_reg_lambda))
            print('Early stop patience: ' + str(self.params.early_stop_patience))
            print('Start learning rate: ' + str(self.params.start_learning_rate))
            print('Log device placement: ' + str(self.params.log_device_placement))
            print('Optimizer: ' + self.params.optimizer)
         
            print('=======================================================\n')

        # set random seed
        self.rng = np.random.RandomState(self.params.init_weights_seed)

        # set pretrained word vector
        self.init_w2v_weight = None
        if not (self.params.pretrain_word2vec_path.lower() == 'none') and not (self.model_exist()):
            print("\nLoding pretrained word vector...")
            self.init_w2v_weight = loader.load_pretrain_word2vec(self.params.pretrain_word2vec_path, self.vocabulary)

            if not (self.params.embedding_size == self.init_w2v_weight.shape[1]):
                print("Convert embedding dimension %d -> %d" % (
                self.params.embedding_size, self.init_w2v_weight.shape[1]))
                self.params.embedding_size = self.init_w2v_weight.shape[1]

            print("")
            
       

        # set learning rate
        default_lr = {'rmsprop': 0.001, 'adadelta': 0.01, 'adam': 0.0001}

        self.optimizer_name = self.params.optimizer
        if self.params.optimizer not in default_lr.keys():
            self.optimizer_name = 'adam'

        self.start_learning_rate = self.params.start_learning_rate
        if self.params.start_learning_rate is None:
            self.start_learning_rate = default_lr[self.optimizer_name]

        # set optimizer
        optims = {
                'rmsprop': tf.train.RMSPropOptimizer,
                'adadelta': tf.train.AdadeltaOptimizer,
                'adam': tf.train.AdamOptimizer
            }

        self.optimizer = optims[self.optimizer_name]

 
    def active_learning(self):
        print('start...')
        print('=======================================================\n')
        #assert self.params.active_learning_method in ['least_confident', 'smallest_margin', 'label_entropy']

        data_dir = self.config['tf_train_data']
        self.params.num_classes = get_tfrecord_classes_counts(self.config['tf_train_data'])
        with tf.Graph().as_default() as g:

            with tf.variable_scope("Input") as scope:

                x_ac, y_ac = inputs(batch_size=self.params.batch_size,
                                    filename=data_dir,
                                    DOC_LEN=self.params.doc_maxlen,
                                    NUM_CLASSES=self.params.num_classes,
                                    shuffle=False, num_epochs=10)

            with tf.variable_scope("DeepModel"):
                # Build VAT model
                vat_model = DeepModel(x_ac, y_ac, None, self.params, init_w2v=self.init_w2v_weight,
                                      dropout_keep_prob=1., word_dropout_keep_prob=1.)

            # define initial operation
            init_global_op = tf.global_variables_initializer()
            init_local_op = tf.local_variables_initializer()

            self.tf_saver = tf.train.Saver(tf.global_variables())

            # Begin Session
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.tf_session = sess

            # run init operation
            sess.run(init_local_op)
            sess.run(init_global_op)

            # restore model
            ckpt_model_name = self.params.model_restore_path + '/best-vat_model.ckpt'
            self.tf_saver = tf.train.Saver(tf.global_variables())
            self.tf_saver.restore(self.tf_session, ckpt_model_name)

            print('\nModel restored')

            # count data size
            num_data_size = get_tfrecord_sample_counts(self.params.active_learning_data_path)
            num_iters = int(np.ceil(float(num_data_size) / self.params.batch_size))

            # self.params.active_learning_frac=0.01
            # num_to_save = int(num_data_size * self.params.active_learning_frac)

            print('Total number of data: ' + str(num_data_size))
            print 'num_classes', self.params.num_classes
            # print('Number to save: '+str(num_to_save))

            # define tfrecord thread
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # active learning methods
            print('\nactive learning methods...')

            tokens = []
            preds = []
            correct_status = []
            labels = []
            tmp = []


            for count in range(num_iters):
            # for count in range(500):
                correct_preds, preds_val, token_ids, label = sess.run(
                    [vat_model.correct_predictions, vat_model.predictions, x_ac, y_ac])
                tokens.extend(token_ids)
                preds.extend(preds_val)
                label_ = [np.argmax(i) for i in label]
                labels.extend(label_)
                correct_status.extend(correct_preds)
                # if count % 100 == 0 and count != 0:
                #     print correct_preds
                # print label_
            
            map_121_to_133 = {}    

            with open('133_to_121.txt', mode='r') as w:
                tmp_ = w.readlines()
                for o in tmp_:
                    o = o.replace('\n', '').split(':')
                    map_121_to_133[o[1]] = o[0]

            map_to_133 = lambda x: map_121_to_133[str(x)]
            
            labels = map(map_to_133, labels)
            preds = map(map_to_133, preds)

            correct_status = [int(status) for status in correct_status]
            acc = float(sum(correct_status)) / len(correct_status)
            print 'final predict acc_rate:', acc

            class_count = np.zeros(shape=[self.params.num_classes], dtype=float)

            # read ind2cat
            print('\nread ind2cat...')

            with open('137int2cat.txt', 'r') as w:
                sum137 = w.readlines()
                names = {}
                for o in sum137:
                    o = o.replace('\n', '').split(',')
                    names[o[0]] = o[1]

            with open('int2label.txt', 'r') as w:
                sum137 = w.readlines()
                ids = {}
                for o in sum137:
                    o = o.replace('\n', '').split(':')
                    ids[o[0]] = o[1]

            print('\nread dict...')
            rev_vocab = {v: k for k, v in self.vocabulary.iteritems()}

            print('\ntext is start...')

            with open(self.params.active_learning_output_path, mode='w') as f:
                texts = ""

                for i in range(num_data_size):
                # for i in range(500):
                    l = str(preds[i])
                    # l = str(labels[i])
                    id1 = str(ids[l]).replace('\r', '')
                    name = names[id1].replace('\r', '')

                    if i % 50 == 0:
                        print('i', i)
                        print('l', l, type(l))
                        # print('pred', str(preds[i]))
                        print('id1', id1, type(id1))
                        print('name', name)

                    text = " ".join([rev_vocab[j] for j in tokens[i]])
                    text = text.replace('<null>', '').strip()
                    # length = len(text)

                    # if length >= self.params.threshold_words:
                    # class_count[preds[i]] += 1

                    texts = str(l) + ',' + str(id1) + ',' + str(name) + ',' + text + '\n'
                    f.write(texts)

            print('\ntext is end...')

            coord.request_stop()
            coord.join(threads)
            sess.close()




    def count_accuracy(self):
        
        print('Input path: '+self.params.count_accuracy_input_tfrecords)
        print('Output path: '+self.params.confusion_matrix_output_dir)
        print('=======================================================\n')
        
        data_dir = self.params.count_accuracy_input_tfrecords
        self.params.num_classes = get_tfrecord_classes_counts(self.config['tf_train_data'])        
        with tf.Graph().as_default() as g:

            with tf.variable_scope('Input') as scope:
                #self.input_y = tf.placeholder(tf.float32, [None, self.params.num_classes], name="input_y")
                
                x_ac, y_ac = inputs(batch_size=self.params.batch_size,
                                    filename=data_dir,
                                    DOC_LEN=self.params.doc_maxlen,
                                    NUM_CLASSES=self.params.num_classes,
                                    shuffle=False, num_epochs=1)

            with tf.variable_scope('DeepModel'):
                # Build VAT model
                vat_model = DeepModel(x_ac, y_ac, None, self.params, init_w2v = self.init_w2v_weight,
                                               dropout_keep_prob = 1., word_dropout_keep_prob = 1.)

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.tf_session = sess

            # self.tf_saver = tf.train.Saver(tf.global_variables())
            self.tf_saver = tf.train.import_meta_graph(self.params.model_restore_path + '/best-vat_model.ckpt.meta')
            self.tf_saver.restore(self.tf_session, tf.train.latest_checkpoint(self.params.model_restore_path))
            #
            # ckpt_model_name = self.params.model_restore_path + '/best-vat_model.ckpt'
            # self.load_model(ckpt_model_name=ckpt_model_name)
            print('\nModel restored')

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            num_data_size = get_tfrecord_sample_counts(data_dir)
            num_iters = int(np.ceil(float(num_data_size) / self.params.batch_size))
            
            print('Total number of data: '+str(num_data_size))
            

            # define tfrecord thread
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
            total_predicts = []
            total_labels = []
            total_tokens = []
            total_scores = []
      
            # for i in range(num_iters):
            for i in range(500):
                predicts, score_val, labels, tokens = sess.run([vat_model.predictions, vat_model.scores, y_ac, x_ac])
                labels = np.argmax(labels, 1)
                print 'preds', predicts
                print 'labels', labels
                
                total_predicts.extend(predicts)
                total_labels.extend(labels)
                total_tokens.extend(tokens)
                total_scores.extend(score_val)

            class_count = np.zeros(shape=[self.params.num_classes], dtype=np.int32)  
            confusion_matrix = np.zeros(shape=[self.params.num_classes, self.params.num_classes], dtype=np.int32)
            
            for i in range(num_data_size):
                class_count[total_labels[i]] += 1
                confusion_matrix[total_labels[i], total_predicts[i]] += 1

            acc_count = np.zeros(shape=[self.params.num_classes], dtype=float)
            for i in range(self.params.num_classes):
                acc_count[i] = confusion_matrix[i, i]
                
            rate = acc_count / class_count
            print 'acc and class', acc_count, class_count
            
            np.savetxt(self.params.confusion_matrix_output_dir, confusion_matrix, fmt='%d', delimiter=',')
            
            np.set_printoptions(threshold=np.nan)
            print 'precision: ', rate
            print 'test case: ', class_count

            # read dict
            rev_vocab = {v: k for k, v in self.vocabulary.iteritems()}

            # output low acc texts.

            if self.params.output_texts_dir != None:
                if not os.path.exists(self.params.output_texts_dir):
                    os.mkdir(self.params.output_texts_dir)


                #read ind2cat 
                with open(self.ind2cat_path, mode='rb') as f:
                    ind2cat = f.readlines()
                names = [re.split(r'(\d+)', i)[4].strip(':').strip() for i in ind2cat]
                
                low_acc_labels = []
                for i, j in enumerate(rate):
                    if j < 0.8:
                        low_acc_labels.append(i)
                        
                print 'low accurate labels'
                for i in low_acc_labels:
                    print i, names[i]
                    
                total_strings = ['' for i in range(self.params.num_classes)]
                wrong_strings = ['' for i in range(self.params.num_classes)]
                for i in range(num_data_size):
                    if total_labels[i] in low_acc_labels:
                        
                        l = str(total_predicts[i]) + '\t'
                        text = ' '.join([rev_vocab[j] for j in total_tokens[i]])
                        text = text.replace('<null>', '').strip()
                        texts = l + text + '\n'
                        total_strings[total_labels[i]] += texts

                        
                        if total_labels[i] != total_predicts[i]:
                            l = str(total_predicts[i]) + '\t'
                            text = ' '.join([rev_vocab[j] for j in total_tokens[i]])
                            text = text.replace('<null>', '').strip()
                            texts = l + text + '\n'
                            wrong_strings[total_labels[i]] += texts
                            
                for i in low_acc_labels:
                    with open(self.params.output_texts_dir+('/whole_label%d.txt' % i), mode='wb') as f:
                        f.write(total_strings[i])
                    with open(self.params.output_texts_dir+('/wrong_label%d.txt' % i), mode='wb') as f:
                        f.write(wrong_strings[i])

            coord.request_stop()
            coord.join(threads)
            sess.close()

            
    def evaluate(self, sess, deepmodel, nbatch):
        correct_list = []
        loss_sum = 0
        for batch_idx in range(nbatch):
            loss, correct = sess.run([deepmodel.loss, deepmodel.correct_predictions])
            print batch_idx,correct
            correct_list += correct.tolist()
            loss_sum += loss

        ndata = len(correct_list)
        accuracy = float(sum(correct_list)) / ndata
        loss = loss_sum / nbatch
        return accuracy, loss

    def save_params(self):
        with open(self.params.model_restore_path + '/params.pkl', 'wb') as f:
            cPickle.dump(self.params, f)

    def save_model(self, aux_name=""):
        if not (aux_name == ""):
            aux_name = aux_name + "-"

        pb_model_name = self.params.model_restore_path + '/' + aux_name + 'model.pb'
        with tf.gfile.FastGFile(pb_model_name, mode='wb') as f:
            f.write(self.output_graph_def.SerializeToString())

        ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'model.ckpt'
        self.tf_saver.save(self.tf_session, ckpt_model_name)
        
    def save_final_model(self):
        saved_model_builder = tf.saved_model.builder.SavedModelBuilder(self.params.model_restore_path+"/"+self.params.cost_type+"-final-model")
        saved_model_builder.add_meta_graph_and_variables(self.tf_session, ["default"])
        saved_model_builder.save()

    def load_model(self, aux_name="", ckpt_model_name=None):
        if not (aux_name == ""):
            aux_name = aux_name + "-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'model.ckpt'

        self.tf_saver.restore(self.tf_session, ckpt_model_name)

    def model_exist(self, aux_name="", ckpt_model_name=None):
        if self.params.model_restore_path.lower() == 'none':
            return False

        if not (aux_name == ""):
            aux_name = aux_name + "-"

        if ckpt_model_name == None:
            ckpt_model_name = self.params.model_restore_path + '/' + aux_name + 'model.ckpt'

        if os.path.exists(ckpt_model_name + '.meta') and os.path.exists(ckpt_model_name + '.index'):
            return True
        else:
            return False







