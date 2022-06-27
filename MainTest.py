import os
import numpy as np
from Readfile import *
from Process import *
import random
random.seed(123)

from absl import app
from absl import flags

import datetime
##############################################################################

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', None, ['Sa-KKH', 'Sauri', 'Sauri-KKH', 'Cj','Nm1','Nm2','Sa','St1'], 'mode of training')
flags.DEFINE_string('filename', None, 'test filename')
flags.DEFINE_integer('fold', 5, 'number of folds')

def main(_):
    data_path = 'dataset/'
    # Sequence, Window, Frequency(Proportion), Wt or Alt, File Name, Sheet Name
    test_param  = {}
    test_param['init'] = 1
    test_param['sheet'] = 'Sheet1'
    modelname = "Model"
    test_param['seq'] = 0
    test_param['mod_seq'] = 1
    test_param['val'] = 2

    if FLAGS.mode in ['Cj']:
        length = 22
        t_length = 4+22+8+3
        bio_num = 275
        modename = "CjCas9"
        
    elif FLAGS.mode in ['efSa']:
        length = 21
        t_length = 34
        bio_num = 263
        modename = "efSaCas9"

    elif FLAGS.mode in ['enCj']:
        length = 22
        t_length = 37
        bio_num = 275
        modename = "enCjCas9"
        
    elif FLAGS.mode in ['eSa']:
        length = 21
        t_length = 34
        bio_num = 263
        modename = "eSaCas9"

    elif FLAGS.mode in ['Nm1']:
        length = 23
        t_length = 4+23+8+3
        bio_num = 287
        modename = "Nm1Cas9"

    elif FLAGS.mode in ['Nm2']:
        length = 23
        t_length = 4+23+7+3
        bio_num = 287
        modename = "Nm2Cas9"

    elif FLAGS.mode in ['Sa']:
        length = 21
        t_length = 4+21+6+3
        bio_num = 263
        modename = "SaCas9"
        
    elif FLAGS.mode in ['Sa-HF']:
        length = 21
        t_length = 34
        bio_num = 263
        modename = "SaCas9-HF"

    elif FLAGS.mode in ['Sa-KKH']:
        length = 21
        t_length = 4+21+6+3
        bio_num = 263
        modename = "SaCas9-KKH"

    elif FLAGS.mode in ['Sa-KKH-HF']:
        length = 21
        t_length = 34
        bio_num = 263
        modename = "SaCas9-KKH-HF"
        
    elif FLAGS.mode in ['Sa-Slug']:
        length = 21
        t_length = 32
        bio_num = 263
        modename = "Sa-SlugCas9"
        
    elif FLAGS.mode in ['Sauri']:
        length = 21
        t_length = 4+21+4+3
        bio_num = 263
        modename = "SauriCas9"

    elif FLAGS.mode in ['Sauri-KKH']:
        length = 21
        t_length = 4+21+4+3
        bio_num = 263
        modename = "SauriCas9-KKH"
        
    elif FLAGS.mode in ['Slug']:
        length = 21
        t_length = 32
        bio_num = 263
        modename = "SlugCas9"
        
    elif FLAGS.mode in ['Slug-HF']:
        length = 21
        t_length = 32
        bio_num = 263
        modename = "SlugCas9-HF"
        
    elif FLAGS.mode in ['sRGN']:
        length = 21
        t_length = 32
        bio_num = 263
        modename = "sRGN3.1"
        
    elif FLAGS.mode in ['St1']:
        length = 19
        t_length = 4+19+6+3
        bio_num = 239
        modename = "St1Cas9"

    else:
        raise NotImplementedError

    test_param['bio']   = test_param['val'] + bio_num
    if FLAGS.filename is None:
        test_param['fname'] = FLAGS.mode+"_sample.xlsx"
    else:
        test_param['fname'] = FLAGS.filename # "FILENAME"

    total_fold = FLAGS.fold

    #TensorFlow config
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    Model = __import__(modelname)
    modulename = FLAGS.mode + "_Test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    if not os.path.exists(modulename):
        os.makedirs(modulename)

    # Initiate Xlsx Output Files
    testbook = xlsxwriter.Workbook(modulename+"/TEST_OUTPUT.xlsx")
    testsheet = [testbook.add_worksheet('{}'.format(test_param['fname'][:-5][:31]))] # -5 is for erasing .xlsx
    test_dict = getfile(data_path, test_param, length, t_length)
    TEST_X = [test_dict['onehot_seq']]
    TEST_mod_X = [test_dict['onehot_mod_seq']]
    TEST_bio = [test_dict['bio']]
    TEST_Y = [test_dict['val']]
    test_row = 0
    test_col = 0

    for seq, mod_seq in zip(test_dict['seq'], test_dict['mod_seq']):
        testsheet[-1].write(test_row, test_col, seq)
        testsheet[-1].write(test_row, test_col+1, mod_seq)
        test_row += 1

    best_model_path_list = [FLAGS.mode+'/Model/{}/best_model'.format(os.listdir(FLAGS.mode+'/Model')[-1])]
    best_model_list                     = []

    for best_model_path in best_model_path_list:
        for best_modelname in os.listdir(best_model_path):
            if "meta" in best_modelname:
                best_model_list.append(best_modelname[:-5])

    print(best_model_list)

    best_model_path = best_model_path_list[0]
    best_model      = best_model_list[0]
    valuelist       = best_model.split('-')
    fulllist        = []

    for value in valuelist:
        try:
            value=int(value)
        except:
            try:    value=float(value)
            except: pass
        fulllist.append(value)

    print(fulllist[2:])

    filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, l_rate, load_episode, node_1, node_2 = fulllist[2:]
    filter_size = [filter_size_1, filter_size_2, filter_size_3]
    filter_num  = [filter_num_1, filter_num_2, filter_num_3]

    args = [filter_size, filter_num, l_rate, 0, None, node_1, node_2]
    # Loading the model with the best validation score and test
    tf.reset_default_graph()
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())
        model = Model.Model(filter_size, filter_num, length, t_length, node_1, node_2, args[2], bio_num)
        saver = tf.train.Saver()
        saver.restore(sess, best_model_path+"/PreTrain-Final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(args[0][0], args[0][1], args[0][2], args[1][0], args[1][1], args[1][2], args[2], load_episode, args[5], args[6]))
        Model_test(sess, TEST_X[0], model, args, load_episode, testbook, testsheet[0], col_index=test_col+2, modulename=modulename, TEST_mod_X=TEST_mod_X[0], TEST_bio=TEST_bio[0], TEST_Y=TEST_Y[0])
        testbook.close()

if __name__ == '__main__':
  app.run(main)
