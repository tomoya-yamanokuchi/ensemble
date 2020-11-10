import time
import pickle
import datetime
import tensorflow as tf



class MYCallBack(tf.keras.callbacks.Callback):
    def __init__(self, learning_info_dir=None):
        self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss = None, None, None, None
        self.now_batch, self.now_epoch = None, None

        self.epochs, self.samples, self.batch_size = None, None, None

        self.learning_info_dir = learning_info_dir
        self.learning_info = dict()
        self.learning_info["loss"] = []
        self.learning_info["acc"]  = []
        self.learning_info["time"] = []

        self.start_run = time.time()


    def print_progress(self):
        epoch = self.now_epoch
        batch = self.now_batch

        epochs = self.epochs
        samples = self.samples
        batch_size = self.batch_size
        sample = batch_size*(batch)

        time_epoch = self.stop_epoch - self.start_epoch
        print_text = "Epoch %d/%d (%d/%d) -- acc: %f loss: %3f " % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss) + "time:" + str(time_epoch)
        print(print_text)

        with open(self.learning_info_dir + "/learning_info.txt", "a") as f:
            f.write(print_text + " \n")

        self.learning_info["loss"].append(self.last_loss)
        self.learning_info["acc"].append(self.last_acc)
        self.learning_info["time"].append(time_epoch)
        with open(self.learning_info_dir + "/learning_info.pickle", "wb") as f:
            pickle.dump(self.learning_info, f)


    # fit開始時
    def on_train_begin(self, logs={}):
        print('\n##### Train Start ##### ' + str(datetime.datetime.now()))

        # パラメータの取得
        self.epochs = self.params['epochs']
        self.samples = self.params['samples']
        self.batch_size = self.params['batch_size']

        # 標準の進捗表示をしないようにする
        self.params['verbose'] = 0


    # batch開始時
    def on_batch_begin(self, batch, logs={}):
        self.now_batch = batch
        

    # batch完了時 (進捗表示)
    def on_batch_end(self, batch, logs={}):
        # 最新情報の更新
        self.last_acc = logs.get('acc') if logs.get('acc') else 0.0
        self.last_loss = logs.get('loss') if logs.get('loss') else 0.0

        # 進捗表示
        # self.print_progress()

    # epoch開始時
    def on_epoch_begin(self, epoch, log={}):
        self.now_epoch   = epoch
        self.start_epoch = time.time()


    # epoch完了時 (進捗表示)
    def on_epoch_end(self, epoch, logs={}):
        # 最新情報の更新
        self.last_val_acc = logs.get('val_acc') if logs.get('val_acc') else 0.0
        self.last_val_loss = logs.get('val_loss') if logs.get('val_loss') else 0.0

        # 進捗表示
        self.stop_epoch = time.time()
        self.print_progress()


    # fit完了時
    def on_train_end(self, logs={}):
        print('\n##### Train Complete ##### ' + str(datetime.datetime.now()))
        
        stop_run = time.time()
        time_run = stop_run - self.start_run
        print("     Timr : " + str(time_run) ) 