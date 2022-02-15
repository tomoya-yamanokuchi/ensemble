from abc import ABCMeta, abstractmethod

'''
    データセットの作成する際にはこのDataseクラス（抽象クラス）を継承してください
'''

class DatasetAbstract(metaclass=ABCMeta):
    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_test(self):
        pass