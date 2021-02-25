import numpy as np
import tensorflow as tf


# this `set_memory_growth` call is needed to run this package inside the
# official tensorflow docker image -- for some reason -- and it isn't
# documented!
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(
        physical_devices[0], enable=True
    )

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)