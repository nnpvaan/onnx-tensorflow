from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

from onnx import defs
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt

import onnx_tf.common as common

class Handler(object):
  """ This class is base handler class.
  Base backend and frontend base handler class inherit this class.

  All operator handler MUST put decorator @onnx_op to register corresponding op.
  """

  ONNX_OP = None

  DOMAIN = defs.ONNX_DOMAIN
  VERSION = 0
  SINCE_VERSION = 0
  PARTIAL_SUPPORT = False
  PS_DESCRIPTION = ''

  @classmethod
  def check_cls(cls):
    if not cls.ONNX_OP:
      common.logger.warning(
          "{} doesn't have ONNX_OP. "
          "Please use Handler.onnx_op decorator to register ONNX_OP.".format(
              cls.__name__))

  @classmethod
  def args_check(cls, node, **kwargs):
    """ Check args. e.g. if shape info is in graph.
    Raise exception if failed.

    :param node: NodeProto for backend.
    :param kwargs: Other args.
    """
    pass

  @classmethod
  def handle(cls, node, **kwargs):
    """ Main method in handler. It will find corresponding versioned handle method,
    whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
    DON'T use it for other purpose.

    :param node: NodeProto for backend.
    :param kwargs: Other args.
    :return: TensorflowNode for backend.
    """
    # Get the version directly to avoid autograph issues with getattr
    version_method_name = "version_{}".format(cls.SINCE_VERSION)
    
    # Use a more explicit approach to avoid autograph transformation issues
    if hasattr(cls, version_method_name):
      cls.args_check(node, **kwargs)
      # Use __dict__ to get the unbound method and call it explicitly
      version_method = cls.__dict__.get(version_method_name)
      if version_method is not None:
        # If it's a classmethod, we need to call it with cls as first argument
        if hasattr(version_method, '__func__'):
          return version_method.__func__(cls, node, **kwargs)
        else:
          # It's already an unbound function
          return version_method(cls, node, **kwargs)
      else:
        # Fallback to getattr if not in __dict__ (inherited methods)
        version_method = getattr(cls, version_method_name)
        return version_method(node, **kwargs)

    raise BackendIsNotSupposedToImplementIt("{} version {} is not implemented.".format(node.op_type, cls.SINCE_VERSION))

  @classmethod
  def get_versions(cls):
    """ Get all support versions.

    :return: Version list.
    """
    versions = []
    for k, v in inspect.getmembers(cls, inspect.ismethod):
      if k.startswith("version_"):
        versions.append(int(k.replace("version_", "")))
    return versions

  @staticmethod
  def onnx_op(op):
    return Handler.property_register("ONNX_OP", op)

  @staticmethod
  def tf_func(func):
    return Handler.property_register("TF_FUNC", func)

  @staticmethod
  def domain(d):
    return Handler.property_register("DOMAIN", d)

  @staticmethod
  def partial_support(ps):
    return Handler.property_register("PARTIAL_SUPPORT", ps)

  @staticmethod
  def ps_description(psd):
    return Handler.property_register("PS_DESCRIPTION", psd)

  @staticmethod
  def property_register(name, value):

    def deco(cls):
      if inspect.isfunction(value) and not common.IS_PYTHON3:
        setattr(cls, name, staticmethod(value))
      else:
        setattr(cls, name, value)
      return cls

    return deco


domain = Handler.domain
onnx_op = Handler.onnx_op
tf_func = Handler.tf_func
partial_support = Handler.partial_support
ps_description = Handler.ps_description
property_register = Handler.property_register
