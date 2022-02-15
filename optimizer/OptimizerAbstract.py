from abc import ABCMeta, abstractmethod

'''
    データセットの作成する際にはこのDataseクラス（抽象クラス）を継承してください
'''

class OptimizerAbstract(metaclass=ABCMeta):
    @abstractmethod
    def get(self, config):
        pass