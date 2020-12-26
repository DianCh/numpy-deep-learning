"""Package that contains all types of layers."""
from .base import Base
from .conv2d import Conv2D, Conv2DThreeFold, Conv2DFourFold
from .linear import Linear
from .pool2d import Pool2D, Pool2DThreeFold, Pool2DFourFold
from .relu import ReLU
from .reshape import Flatten, Squeeze2D
from .sigmoid import Sigmoid
