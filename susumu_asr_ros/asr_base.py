"""ASR抽象基底クラス."""
from abc import ABC, abstractmethod


class ASRBase(ABC):

    @abstractmethod
    def run(self):
        pass
