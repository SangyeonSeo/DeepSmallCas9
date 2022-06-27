import numpy as np
import pyexcel as pe
from random import shuffle
np.random.seed(1)
##############################################################################

# One hot encoding for DNA Sequence
def preprocess_seq(data, length):
    print("Start preprocessing the sequence done 2d")

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=int)
    print(np.shape(data), len(data), length)
    print(data)
    for l in range(len(data)):
        for i in range(length):

            try: data[l][i]
            except: print(data[l], len(data[l]), i, length, len(data))

            if data[l][i] in "Aa":   DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc": DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg": DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt": DATA_X[l, 0, i, 3] = 1
            elif data[l][i] in "Xx": pass # remain 0
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()
        #loop end: i
    #loop end: l
    print("Preprocessing the sequence done")
    return DATA_X
#def end: preprocess_seq

# init, sheet, fname, prev_seq, mod_seq, val, fold_row, seq
def getfile(path, param, length, t_length):
    seq_dict = {}

    seq_keys = [*param.keys()]
    for key in ['init', 'sheet', 'fname']:
        seq_keys.remove(key)

    assert 'seq' == seq_keys[0]

    xlsx = pe.get_book(file_name=path+param['fname'])[0]

    for seq_key in seq_keys:
        if seq_key == 'bio':
            bio = []
            for idx in range(param['val']+1, param['bio']+1):
                bio.append(np.array(xlsx.column[idx][param['init']:]))
            seq_dict[seq_key] = np.swapaxes(np.array(bio), 0, 1)
        else:
            seq_dict[seq_key] = np.array(xlsx.column[param[seq_key]][param['init']:])

    index = list(range(len(seq_dict['seq'])))
    index = [index for index, value in enumerate(seq_dict['seq']) if value == ('')]
    index = list(set(range(len(seq_dict['seq']))) - set(index))

    shuffle(index)

    for seq_key in seq_keys:
        seq_dict[seq_key] = seq_dict[seq_key][index]

    if 'val' in seq_dict.keys():
        seq_dict['val'] = (np.fromiter(map(float, seq_dict['val']), dtype=np.float32))+0.0001 # to prevent 0
        if np.isnan(np.sum(seq_dict['val'])) == True:
            print(seq_dict['val'])
            raise ValueError
        seq_dict['val'] = np.expand_dims(seq_dict['val'], axis=-1)
    if 'fold_row' in seq_dict.keys():
        for fold_val in seq_dict['fold_row']:
            if type(fold_val) != np.int64:
                print(fold_val)
                import ipdb;ipdb.set_trace()
            else:
                pass #print("success")
        seq_dict['fold_row'] = (np.fromiter(map(int, seq_dict['fold_row']), dtype=np.uint8))

    for seq_key in seq_keys:
        seq_dict['onehot_seq'] = preprocess_seq(seq_dict['seq'], length)
        seq_dict['onehot_mod_seq'] = preprocess_seq(seq_dict['mod_seq'], t_length)

    return seq_dict