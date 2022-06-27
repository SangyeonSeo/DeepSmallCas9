import tensorflow as tf
from numpy import *
import scipy.stats
import math
from Readfile import *
import xlsxwriter
tf.random.set_random_seed(seed=1)
random.seed(1)
##############################################################################

# Test Model
def Model_test(sess, TEST_X, model, args, load_episode, testvalbook, testvalsheet, col_index=3, modulename="", TEST_Y=None, TEST_mod_X=None, TEST_bio=None):
    test_batch = 100
    TEST_Z = zeros((TEST_X.shape[0], 1), dtype=float)

    for i in range(int(ceil(float(TEST_X.shape[0])/float(test_batch)))):
        if np.all(TEST_mod_X == None) and np.all(TEST_bio == None):
            raise ValueError
            Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch],
                    model.is_training: False}
        elif not np.all(TEST_mod_X == None) and np.all(TEST_bio == None):
            raise ValueError
            Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch],
                    model.mod_inputs: TEST_mod_X[i*test_batch:(i+1)*test_batch],
                    model.is_training: False}
        elif np.all(TEST_mod_X == None) and not np.all(TEST_bio == None):
            raise ValueError
            Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch],
                    model.bios: TEST_bio[i*test_batch:(i+1)*test_batch],
                    model.is_training: False}
        else:
            Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch],
                    model.mod_inputs: TEST_mod_X[i*test_batch:(i+1)*test_batch],
                    model.bios: TEST_bio[i*test_batch:(i+1)*test_batch],
                    model.is_training: False}

        TEST_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]

    print(TEST_Z[0], np.sum(TEST_Z))
    if TEST_Y is not None:
        spearman = 0
        pearson = 0

    testval_row = 0
    for test_value in (TEST_Z):
        if math.isnan(test_value[0])==True:
            test_value = 0
            print("Warning! - testvalue is Nan")
        testvalsheet.write(testval_row, col_index, test_value)
        testval_row += 1
    if TEST_Y is not None:
        col_index += 1
        testval_row = 0
        testvalsheet.write(testval_row, col_index, "Spearman")
        testval_row += 1
        testvalsheet.write(testval_row, col_index, "Pearson")

        col_index += 1
        testval_row = 0
        testvalsheet.write(testval_row, col_index, spearman)
        testval_row += 1
        testvalsheet.write(testval_row, col_index, pearson)
    return

# Load model
def Model_Load(sess, filter_size, filter_num, length, t_length, node_1, node_2, args, load_episode, bio_num, ver="Complex", modelname="Model_1layer", modulename=""):
    Model = __import__(modelname)
    model = Model.Model(filter_size, filter_num, length, t_length, node_1, node_2, args[2], bio_num)
    saver = tf.train.Saver()
    saver.restore(sess, modulename+"model_checkpoints/PreTrain-Final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
            args[0][0], args[0][1], args[0][2], args[1][0], args[1][1], args[1][2], args[2], load_episode, args[5], args[6]))
    return model
