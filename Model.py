import tensorflow as tf
from numpy import *
from Readfile import *
##############################################################################

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, stride_shape, name, is_training):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                        num_filters]
    weights   = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                        name=name+'_W')
    bias      = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1, stride_shape[0], stride_shape[1], 1], padding='VALID')
    out_layer += bias
    out_layer = tf.layers.dropout(tf.nn.relu(out_layer), 0.3, is_training)
    return out_layer

class Model(object):
    def __init__(self, filter_size, filter_num, length, t_length, node_1 = 80, node_2 = 60, l_rate = 0.005, bio_num=0):
        self.inputs         = tf.placeholder(tf.float32, [None, 1, length, 4])
        self.mod_inputs     = tf.placeholder(tf.float32, [None, 1, t_length, 4])
        self.bios           = tf.placeholder(tf.float32, [None, bio_num])
        self.targets        = tf.placeholder(tf.float32, [None, 1])
        self.is_training    = tf.placeholder(tf.bool)

        L_filter_num = 4
        stride = 1
        if filter_num[0] == 0:
            raise NotImplementedError
        else:
            L_pool_0 = create_new_conv_layer(self.inputs, L_filter_num, filter_num[0]*3, [1, filter_size[0]], [1, stride], name='conv1', is_training=self.is_training)
            L_pool_1 = create_new_conv_layer(self.mod_inputs, L_filter_num, filter_num[0]*3, [1, filter_size[0]], [1, stride], name='conv2', is_training=self.is_training)

        with tf.variable_scope('Fully_Connected_Layer1'):
            layer_node_0 = int((length-filter_size[0])/stride)+1
            node_num_0   = layer_node_0*filter_num[0]*3
            L_flatten_0  = tf.reshape(L_pool_0, [-1, node_num_0])
            layer_node_1 = int((t_length-filter_size[0])/stride)+1
            node_num_1   = layer_node_1*filter_num[0]*3
            L_flatten_1  = tf.reshape(L_pool_1, [-1, node_num_1])
            
            L_flatten_concat = tf.concat([L_flatten_0, L_flatten_1, self.bios], 1, name='concat')

            W_fcl1       = tf.get_variable("W_fcl1", shape=[node_num_0+node_num_1+bio_num, node_1])
            B_fcl1       = tf.get_variable("B_fcl1", shape=[node_1])
            L_fcl1_pre   = tf.nn.bias_add(tf.matmul(L_flatten_concat, W_fcl1), B_fcl1)
            L_fcl1       = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop  = tf.layers.dropout(L_fcl1, 0.3, self.is_training)

        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2       = tf.get_variable("W_fcl2", shape=[node_1, node_2])
            B_fcl2       = tf.get_variable("B_fcl2", shape=[node_2])
            L_fcl2_pre   = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2       = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop  = tf.layers.dropout(L_fcl2, 0.3, self.is_training)

        with tf.variable_scope('Output_Layer'):
            W_out        = tf.get_variable("W_out", shape=[node_2, 1])
            B_out        = tf.get_variable("B_out", shape=[1])
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss    = tf.reduce_mean(tf.square(self.targets - self.outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        self.gvs = optimizer.compute_gradients(self.obj_loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)