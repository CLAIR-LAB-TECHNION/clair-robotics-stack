from abc import ABC, abstractmethod


class ThreeLayerStateEstimator(ABC):
    @abstractmethod
    def estimate_state(self, obs: dict) -> tuple:
        pass
