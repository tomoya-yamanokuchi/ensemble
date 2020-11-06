
import tensorflow as tf
import time
import os
import json


def reload_config(FLAGS):
    # If we are reloading a model, overwrite the flags
    if FLAGS.reload_model is not '':
        with open('%s/%s' % (os.path.dirname(FLAGS.reload_model), 'config.json')) as data_file:
            config_dict = json.load(data_file)

        for key, value in list(config_dict.items()):
            attr_remove = ['run_name', 'log_dir']

            if key not in attr_remove:
                FLAGS.__setattr__(key, value)

    return FLAGS


def get_image_config():
    cl = tf.app.flags

    flags_dict = tf.flags.FLAGS._flags()
    keys_list = [keys for keys in flags_dict]

    if 'gpu' not in keys_list: 
        cl.DEFINE_string('gpu', '1', 'Comma seperated list of GPUs')
    if 'dataset' not in keys_list: 
        cl.DEFINE_string('dataset', '/home/dl-box/jst/python_code/drkvae/master/logs/plane_64x64_N2800_seq10_20201029133630_20201029141634_kvae/1step_prediction_error_data/pred_error_from_random.npz', 'dataset')
    
    # feature transformation 
    cl.DEFINE_float('scale_inputs',  100,   'scale_inputs')
    cl.DEFINE_float('bias',         0,   'bias') # 1e-7
 
    # network setting
    cl.DEFINE_integer('dim_inputs',    5,   'dim_inputs')
    cl.DEFINE_integer('dim_outputs',   1,   'dim_outputs')
    cl.DEFINE_integer('N_ensemble',    1,   'N_ensemble')
    cl.DEFINE_string('units',  '512, 512, 512', 'units')
    # cl.DEFINE_string('units',  '200, 200, 200, 200', 'units')

    # optimizer setting
    cl.DEFINE_integer('epoch',        1000,   'epoch')
    if 'batch_size' not in keys_list: 
        cl.DEFINE_integer('batch_size',    32,    'batch_size')
    if 'learning_rate' not in keys_list: 
        cl.DEFINE_float('learning_rate', 0.001, 'learning_rate')

    if 'reload_model' not in keys_list: 
        cl.DEFINE_string('reload_model', '', 'Path to the model.ckpt file')
    
    if 'log_dir' not in keys_list: 
        cl.DEFINE_string('log_dir',  'logs', 'Directory to save files in')

    print((3))

    return cl


if __name__ == '__main__':
    config = get_image_config()
    config.DEFINE_bool('test', True, 'test')
    config = reload_config(config.FLAGS)

    print((config.dataset))
    config.dataset = 'test'
    print((config.dataset))

