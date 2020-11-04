
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
            attr_remove = ['gpu', 'run_name', 'log_dir', 'n_steps_gen', 'reload_model', 'reload_vae_model',
                           'display_step', 'generate_step', 'init_lr', 'decay_steps',
                           'dim_aux', 't_init_mask_image', 't_steps_mask_image', 'test_mask_scope', 
                           'test_miss_prob', 'dataset_test', 'dataset_eval','t_init_mask', 'batch_size', 'num_series', 
                           'use_all_testdata', 'test_data_type', 'scale_elbo_kf', 'dataset_adapt', 
                           'init_lr_adapt', 'num_epochs_adapt', 'noise_transition', 'noise_emission', 'use_vae', 
                           'many_to_one', 'common_init_dpn_param', 'max_grad_norm_adapt', 'init_cov', 
                           'horizon', 'N_sample_cem', 'N_elite','max_iters','upper_bound','lower_bound',
                           'init_mean','init_var','epsilon','alpha_mean','alpha_var']

            if key not in attr_remove:
                FLAGS.__setattr__(key, value)

    return FLAGS


def get_image_config():
    cl = tf.app.flags

    cl.DEFINE_integer('seed', 3,  'seed num')
    cl.DEFINE_string('gpu', '1', 'Comma seperated list of GPUs')

    cl.DEFINE_string('dataset', '/home/dl-box/jst/python_code/drkvae/master/logs/plane_64x64_N2800_seq10_20201029133630_20201029141634_kvae/1step_prediction_error_data/pred_error_from_random.npz', 'dataset')
    
    # DNN setting
    # cl.DEFINE_float('scale_inputs',  1e4, 'scale_inputs')
    cl.DEFINE_integer('dim_inputs',    1,   'dim_inputs')
    cl.DEFINE_integer('dim_outputs',   1,   'dim_outputs')
    cl.DEFINE_integer('N_ensemble',    5,   'N_ensemble')
    

    cl.DEFINE_string('reload_model', '', 'Path to the model.ckpt file')
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

