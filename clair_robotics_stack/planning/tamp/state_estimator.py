from abc import ABC, abstractmethod


class ThreeLayerStateEstimator(ABC):
    @abstractmethod
    def estimate_task_state(self, obs: dict):
        pass

    @abstractmethod
    def estimate_motion_state(self, obs: dict):
        pass

    @abstractmethod
    def estimate_control_state(self, obs: dict):
        pass
