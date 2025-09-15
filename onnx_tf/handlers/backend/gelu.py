import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Gelu")
class Gelu(BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        approximate = node.attrs.get("approximate", "none")

        if approximate == "tanh":
            # Tanh approximation:
            # y = 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)
            coeff = tf.constant(0.044715, dtype=x.dtype)
            inner = sqrt_2_over_pi * (x + coeff * tf.pow(x, 3))
            return [0.5 * x * (1.0 + tf.tanh(inner))]
        else:
            # Exact: y = 0.5 * x * (1 + erf(x/sqrt(2)))
            sqrt_2 = tf.constant(1.4142135623730951, dtype=x.dtype)
            return [0.5 * x * (1.0 + tf.math.erf(x / sqrt_2))]

    @classmethod
    def version_20(cls, node, **kwargs):
        return cls._common(node, **kwargs)
