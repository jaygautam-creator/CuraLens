import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

class PatchedBatchNormalization(BatchNormalization):
    """
    Keras 2.13+ expects 'axis' to be an integer (or a single-element list/tuple).
    Older versions sometimes saved it as a list like [3].
    This patch ensures it's converted to an integer if necessary.
    """
    def __init__(self, **kwargs):
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            kwargs['axis'] = kwargs['axis'][0]
        super().__init__(**kwargs)

def safe_load_model(path, compile=False):
    """
    Utility to load Keras models with the BatchNormalization patch.
    """
    custom_objects = {'BatchNormalization': PatchedBatchNormalization}
    return tf.keras.models.load_model(
        path, 
        compile=compile, 
        custom_objects=custom_objects
    )
