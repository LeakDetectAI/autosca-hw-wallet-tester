import tensorflow as tf

import deepscapy

# import keras
# import csrank as cs
devices = tf.config.list_physical_devices()
print(devices)
print(tf.config.list_physical_devices("GPU"))

# print('version', keras.__version__)
# print('path', keras.__path__)
print('tf version', tf.__version__)
print('tf path', tf.__path__)
print('tf path', deepscapy.__path__)
print('devices', [device.name for device in devices])
