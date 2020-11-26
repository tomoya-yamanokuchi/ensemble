
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


def cl_define_with_key_check(tf_app_flags, keys_list, define_type, define_name, default_item):
    if   define_type == "string" :  
        if define_name not in keys_list: 
            tf_app_flags.DEFINE_string(define_name, default_item, define_name)
    elif define_type == "integer":  
        if define_name not in keys_list: 
            tf_app_flags.DEFINE_integer(define_name, default_item, define_name)
    elif define_type == "float"  :  
        if define_name not in keys_list:
            tf_app_flags.DEFINE_float(define_name, default_item, define_name)


def get_image_config():
    cl = tf.app.flags

    flags_dict = tf.flags.FLAGS._flags()
    keys_list = [keys for keys in flags_dict]

    cl_define_with_key_check(cl, keys_list, "string", "gpu", "0")

    kvae_model = "seesaw_64x64_N5000_seq30_cem_random_mixed_20201120205316_kvae"
    cl_define_with_key_check(cl, keys_list, "string", "kvae_model", kvae_model)
    cl_define_with_key_check(cl, keys_list, "string", "dataset",    "/hdd_mount/logs/" + kvae_model + "/1step_prediction_error_data/pred_error_from_canonical.npz")

    # feature transformation 
    cl_define_with_key_check(cl, keys_list, "float", "scale_inputs", 100)
    cl_define_with_key_check(cl, keys_list, "float", "bias", 0)
 
    # network setting
    cl_define_with_key_check(cl, keys_list, "integer", "dim_inputs", 15)
    cl_define_with_key_check(cl, keys_list, "integer", "dim_outputs", 1)
    cl_define_with_key_check(cl, keys_list, "integer", "N_ensemble", 5)
    cl_define_with_key_check(cl, keys_list, "string",  "units", '512, 512, 512')

    # optimizer setting
    cl_define_with_key_check(cl, keys_list, "integer", "epoch", 500)
    cl_define_with_key_check(cl, keys_list, "integer", "batch_size", 64)
    cl_define_with_key_check(cl, keys_list, "float",   "learning_rate", 0.001)
    cl_define_with_key_check(cl, keys_list, "string",  "reload_model", '')
    cl_define_with_key_check(cl, keys_list, "string",  "log_dir", 'logs')

    print((3))

    return cl


if __name__ == '__main__':
    config = get_image_config()
    config.DEFINE_bool('test', True, 'test')
    config = reload_config(config.FLAGS)

    print((config.dataset))
    config.dataset = 'test'
    print((config.dataset))

