"""Implements learning rate schedulers."""
from abc import ABC, abstractmethod


class Scheduler(ABC):
    """Generic scheduler."""

    @abstractmethod
    def step(self):
        """Make one step to adjust the learning rate."""
        pass


class StepLR(Scheduler):
    """Exponentially decrease the optimizer's learning rate every fixed
    number of steps.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        pass

    def step(self):
        pass


class MultiStepLR(Scheduler):
    """Exponetially decrease the optimizer's learning rate at specified
    steps.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        pass

    def step(self):
        pass
