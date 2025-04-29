from abc import ABC, abstractmethod

class BankEmailParser(ABC):
    @abstractmethod
    def parse(self, message: dict):
        """Return a list of Transaction objects extracted from this message."""
        pass