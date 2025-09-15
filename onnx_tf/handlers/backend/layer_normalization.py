import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("LayerNormalization")
@tf_func(tf.nn.l2_normalize)
class LayerNormalization(BackendHandler):

    @classmethod
    def version_17(cls, node, **kwargs):
        """
        LayerNormalization as described in https://arxiv.org/abs/1607.06450.

        The overall computation can be split into two stages:
        1. Standardization: Makes normalized elements have zero mean and
           unit variances
        2. Scaling: Scales and shifts the results of the first stage

        For standardization:
        Mean = ReduceMean<axes=normalized_axes>(X)
        D = Sub(X, Mean)
        DD = Mul(D, D)
        Var = ReduceMean<axes=normalized_axes>(DD)
        VarEps = Add(Var, epsilon)
        StdDev = Sqrt(VarEps)
        InvStdDev = Reciprocal(StdDev)
        Normalized = Mul(D, InvStdDev)

        For scaling:
        NormalizedScaled = Mul(Normalized, Scale)
        Y = Add(NormalizedScaled, B)
        """
        tensor_dict = kwargs["tensor_dict"]

        # Get inputs
        x = tensor_dict[node.inputs[0]]
        scale = tensor_dict[node.inputs[1]]
        bias = tensor_dict[node.inputs[2]] if len(node.inputs) > 2 else None

        # Get attributes
        axis = node.attrs.get("axis", -1)
        epsilon = node.attrs.get("epsilon", 1e-5)
        stash_type = node.attrs.get("stash_type", 1)

        # Get input rank
        x_rank = len(x.shape)

        # Convert negative axis to positive
        if axis < 0:
            axis = x_rank + axis

        # Determine the normalized axes - from axis to the last dimension
        normalized_axes = list(range(axis, x_rank))

        # Cast to computation precision if needed (stash_type=1 means float32)
        if stash_type == 1:
            computation_dtype = tf.float32
            x_compute = tf.cast(x, computation_dtype)
            scale_compute = tf.cast(scale, computation_dtype)
            if bias is not None:
                bias_compute = tf.cast(bias, computation_dtype)
        else:
            computation_dtype = x.dtype
            x_compute = x
            scale_compute = scale
            if bias is not None:
                bias_compute = bias

        # Stage 1: Standardization
        # Compute mean along normalized axes
        mean = tf.reduce_mean(x_compute, axis=normalized_axes, keepdims=True)

        # Compute centered values
        centered = tf.subtract(x_compute, mean)

        # Compute variance
        squared_diff = tf.square(centered)
        variance = tf.reduce_mean(squared_diff, axis=normalized_axes,
                                  keepdims=True)

        # Add epsilon to variance
        variance_eps = tf.add(variance, epsilon)

        # Compute standard deviation and its reciprocal
        std_dev = tf.sqrt(variance_eps)
        inv_std_dev = tf.reciprocal(std_dev)

        # Normalize
        normalized = tf.multiply(centered, inv_std_dev)

        # Stage 2: Scale and shift
        # Scale
        scaled = tf.multiply(normalized, scale_compute)

        # Add bias if provided
        if bias is not None:
            result = tf.add(scaled, bias_compute)
        else:
            result = scaled

        # Cast back to original dtype if needed
        if stash_type == 1 and x.dtype != computation_dtype:
            result = tf.cast(result, x.dtype)

        # Prepare outputs
        outputs = [result]

        # Add optional mean and inv_std_dev outputs if requested
        if len(node.outputs) > 1:
            # Mean output (cast back to original dtype if needed)
            if stash_type == 1 and x.dtype != computation_dtype:
                mean_output = tf.cast(mean, x.dtype)
            else:
                mean_output = mean
            outputs.append(mean_output)

        if len(node.outputs) > 2:
            # InvStdDev output (cast back to original dtype if needed)
            if stash_type == 1 and x.dtype != computation_dtype:
                inv_std_dev_output = tf.cast(inv_std_dev, x.dtype)
            else:
                inv_std_dev_output = inv_std_dev
            outputs.append(inv_std_dev_output)

        return outputs