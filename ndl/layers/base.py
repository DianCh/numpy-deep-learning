"""A Base class for learnable layers."""
import numpy as np

from ndl.utils import calculate_fan


class Base:
    """Base class for learnable layers with weights and biases."""

    def clear_gradients(self):
        self.dW *= 0.0
        self.db *= 0.0

    def kaiming_uniform_init_weights(self, a=0, fan_mode="in"):
        assert fan_mode in ("in", "out"), "Invalid fan mode."
        fan_in, fan_out = calculate_fan(self.W)
        gain = np.sqrt(2.0)
        std = (
            gain / np.sqrt(fan_in)
            if fan_mode == "in"
            else gain / np.sqrt(fan_out)
        )
        bound = np.sqrt(3.0) * std
        self.W[:] = np.random.uniform(-bound, bound, self.W.shape)

    def kaiming_uniform_init_biases(self):
        fan_in, _ = calculate_fan(self.W)
        bound = 1 / np.sqrt(fan_in)
        self.b[:] = np.random.uniform(-bound, bound, self.b.shape)
