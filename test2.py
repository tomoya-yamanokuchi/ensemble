def LoadNetwork(self, model_num = -1, value_flag = True, discriminator_flag = True):
    if value_flag:
        checkpoint_value = tf.train.get_checkpoint_state(self.save_network_path_ + '/Value/')
        if checkpoint_value and checkpoint_value.model_checkpoint_path:
            if model_num == -1:
                path_value = checkpoint_value.model_checkpoint_path
            elif model_num.isdigit():
                path_value = self.MakePath(checkpoint_value.model_checkpoint_path, int(model_num))
            else:
                path_value = model_num
            self.agent_.updater_.saver.restore(self.agent_.updater_.sess, path_value)
            print('Successfully loaded: ' + path_value)
        else:
            print('Load failed')

    if discriminator_flag:
        checkpoint_discriminator = tf.train.get_checkpoint_state(self.save_network_path_ + '/Discriminator/')
        if checkpoint_discriminator and checkpoint_discriminator.model_checkpoint_path:
            if model_num == -1:
                path_discriminator = checkpoint_discriminator.model_checkpoint_path
            elif model_num.isdigit():
                path_discriminator = self.MakePath(checkpoint_discriminator.model_checkpoint_path, int(model_num))
            else:
                path_discriminator = model_num
            self.reward_func.saver.restore(self.agent_.updater_.sess, path_discriminator)
            print('Successfully loaded: ' + path_discriminator)
        else:
            print('Load failed')