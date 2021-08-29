# 将不同长度的答案拓展到相同长度，以便训练模型
import numpy as np


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size # 要填充到size大小
        self.pad = pad   # 填充字符的index

    # 特殊方法__call__(),可以把一个类实例变成一个可调用对象
    # 例如 l = LabelTransform(size, pad), l(label)
    
    # numpy.pad函数（https://numpy.org/doc/stable/reference/generated/numpy.pad.html）
    # numpy.pad(array, pad_width, mode='constant', **kwargs)
    # array:The array to pad
    # mode:‘constant’ (default),Pads with a constant value.
    
    def __call__(self, label):
        label = np.pad(label,
                        (0, (self.size - label.shape[0])),
                        mode='constant',
                        constant_values=self.pad)
        return label
