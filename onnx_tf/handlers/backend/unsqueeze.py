import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Unsqueeze")
@tf_func(tf.expand_dims)
class Unsqueeze(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    axes = attrs.pop("axes")
    if len(axes) != 1:
      x = kwargs["tensor_dict"][node.inputs[0]]
      for axis in sorted(axes):
        x = tf.expand_dims(x, axis=axis)
      return [x]
    attrs["axis"] = axes[0]
    return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    axes = kwargs["tensor_dict"][node.inputs[1]]
    axes = tf.sort(axes)
    x = kwargs["tensor_dict"][node.inputs[0]]

    def apply_unsqueeze():
      # Use tf.expand_dims iteratively for multiple axes
      result = x
      # Process axes in reverse order to maintain correct indexing
      axes_reversed = tf.reverse(axes, [0])
      
      def body(i, tensor):
        axis = axes_reversed[i]
        expanded = tf.expand_dims(tensor, axis=axis)
        return i + 1, expanded
      
      def condition(i, tensor):
        return i < tf.size(axes)
      
      _, result = tf.while_loop(
        condition, body,
        [0, result],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape(None)]
      )
      return result

    def no_unsqueeze():
      return cls.make_tensor_from_onnx_node(node, **kwargs)[0]

    # Use tf.cond instead of Python if
    result = tf.cond(
      tf.size(axes) > 0,
      apply_unsqueeze,
      no_unsqueeze
    )

    return [result]

  @classmethod
  def version_21(cls, node, **kwargs):
    return cls.version_13(node, **kwargs)
