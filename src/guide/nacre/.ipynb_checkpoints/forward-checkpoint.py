"""Keras LSTM MC-dropout surrogate and normalized-strain RBF covariance closure.

TensorFlow is imported only when loading or evaluating the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from guide.types import PredictiveDistribution

from .representation import NacreRepresentation


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Nacre LSTM operations require TensorFlow. Install the nacre extra: pip install -e '.[nacre]'.") from exc
    return tf


def load_nacre_forward_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing nacre forward checkpoint: {path}.")
    tf = require_tensorflow()
    model = tf.keras.models.load_model(str(path), compile=False)
    if model.input_shape[-1] != 11 or model.output_shape[-1] != 1:
        raise ValueError("Nacre model must accept eleven features and output one stress per timestep.")
    return model


def kernel_covariance(std, model_strains, *, length_scale=.35, jitter=.05,
                      std_scale=1.):
    """Return ``s² D K D + jitter I``; coordinates are standardized strain.

    ``std_scale`` scales dropout standard deviations before adding jitter.
    """
    values = np.asarray(std, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    grid = np.asarray(model_strains, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(grid):
        raise ValueError("std must have shape (n_designs, n_response_points).")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.all(np.isfinite(grid)):
        raise ValueError("Standard deviations and strain coordinates must be finite; std must be nonnegative.")
    if not all(np.isfinite(v) for v in (length_scale, jitter, std_scale)):
        raise ValueError("Covariance parameters must be finite.")
    if length_scale <= 0 or jitter < 0 or std_scale <= 0:
        raise ValueError("length_scale and std_scale must be positive; jitter cannot be negative.")
    rbf = np.exp(-.5 * ((grid[:, None] - grid[None, :]) / length_scale) ** 2)
    scaled = values * std_scale
    covariance = scaled[:, :, None] * rbf[None] * scaled[:, None, :]
    covariance[:, np.arange(len(grid)), np.arange(len(grid))] += jitter
    return covariance


def clone_with_fixed_dropout(model, *, mc_samples=30, seed=0):
    """Replace dense Dropout layers by a fixed, batch-invariant MC mask bank.

    A design is repeated ``mc_samples`` consecutive times. Every design sees the
    same fixed draws, preserving a deterministic approximate likelihood for MH.
    """
    tf = require_tensorflow()
    if mc_samples < 2:
        raise ValueError("mc_samples must be at least 2.")
    count = 0

    class FixedMCDropout(tf.keras.layers.Layer):
        def __init__(self, rate, layer_seed, **kwargs):
            super().__init__(**kwargs)
            self.rate = float(rate)
            self.layer_seed = int(layer_seed)

        def call(self, inputs, training=None):
            shape = tf.shape(inputs)
            tf.debugging.assert_equal(shape[0] % mc_samples, 0,
                                      message="Fixed MC batch must be a multiple of mc_samples.")
            bank_shape = tf.concat([[1, mc_samples], shape[1:]], axis=0)
            random = tf.random.stateless_uniform(bank_shape, seed=(seed, self.layer_seed), dtype=inputs.dtype)
            mask = tf.cast(random >= self.rate, inputs.dtype) / (1. - self.rate)
            grouped = tf.reshape(inputs, tf.concat([[-1, mc_samples], shape[1:]], axis=0))
            return tf.reshape(grouped * mask, shape)

    def clone(layer):
        nonlocal count
        if isinstance(layer, tf.keras.layers.Dropout):
            if layer.noise_shape is not None:
                raise ValueError("Fixed MC mode expects ordinary elementwise Dropout (noise_shape=None).")
            count += 1
            return FixedMCDropout(layer.rate, count, name=layer.name, dtype=layer.dtype)
        if isinstance(layer, tf.keras.layers.RNN):
            cell = layer.cell
            if getattr(cell, "dropout", 0) or getattr(cell, "recurrent_dropout", 0):
                raise ValueError("Fixed MC mode does not replace internal recurrent dropout; use the supplied architecture with external Dropout layers.")
        if isinstance(layer, tf.keras.Model):
            raise ValueError("Nested Keras models need an explicit fixed-dropout adapter.")
        return layer.__class__.from_config(layer.get_config())

    fixed = tf.keras.models.clone_model(model, clone_function=clone)
    if count == 0:
        raise ValueError("MC-dropout model contains no supported Dropout layers.")
    fixed.set_weights(model.get_weights())
    return fixed


class NacreMCDropoutForwardModel:
    """Adapt a frozen Keras surrogate to GUIDe's probabilistic forward interface."""

    def __init__(self, model, representation: NacreRepresentation, *, mc_samples=30,
                 std_scale=0.74, kernel_length_scale=.35, jitter=.05,
                 dropout_mode="fixed", seed=0, batch_size=8):
        if mc_samples < 2 or batch_size < 1:
            raise ValueError("mc_samples must be >=2 and batch_size must be positive.")
        if dropout_mode not in ("fixed", "stochastic"):
            raise ValueError("dropout_mode must be 'fixed' or 'stochastic'.")
        if std_scale is None:
            raise ValueError("Nacre std_scale must be specified.")
        self.model, self.representation = model, representation
        self.mc_samples, self.batch_size = int(mc_samples), int(batch_size)
        self.std_scale, self.kernel_length_scale, self.jitter = float(std_scale), float(kernel_length_scale), float(jitter)
        self.dropout_mode, self.seed = dropout_mode, int(seed)
        kernel_covariance(np.zeros(representation.response_dimension), representation.model_strains,
                          length_scale=self.kernel_length_scale, jitter=self.jitter,
                          std_scale=self.std_scale)
        self._mc_model = clone_with_fixed_dropout(model, mc_samples=self.mc_samples, seed=self.seed) if dropout_mode == "fixed" else model
        self._mc_predict = self._compile_mc_predictor()

    def _compile_mc_predictor(self):
        """Share deterministic prefix work across MC draws, preserving all weights.

        The supplied checkpoint has no recurrent dropout: both LSTMs and the
        first dense layer therefore give identical activations for every draw.
        Repeat those activations only at the first external Dropout. Unsupported
        graphs fall back to evaluating the complete model for every draw.
        """
        tf = require_tensorflow()
        prefix, tail = None, self._mc_model
        previous = self._mc_model.inputs[0]
        for original, layer in zip(self.model.layers, self._mc_model.layers, strict=True):
            if isinstance(original, tf.keras.layers.InputLayer):
                continue
            if isinstance(original, tf.keras.layers.Dropout):
                if layer.input is previous:
                    try:
                        prefix = tf.keras.Model(self._mc_model.inputs, layer.input)
                        tail = tf.keras.Model(layer.input, self._mc_model.output)
                    except ValueError:
                        prefix, tail = None, self._mc_model
                break
            if not isinstance(original, (tf.keras.layers.Dense, tf.keras.layers.LSTM,
                                         tf.keras.layers.Activation)) or layer.input is not previous:
                break
            if isinstance(original, tf.keras.layers.LSTM) and (
                original.stateful or original.dropout or original.recurrent_dropout
            ):
                break
            previous = layer.output

        @tf.function(input_signature=[tf.TensorSpec(
            [None, self.representation.response_dimension, 11], tf.float32
        )], autograph=False, reduce_retracing=True)
        def predict(inputs):
            features = inputs if prefix is None else prefix(inputs, training=False)
            repeated = tf.repeat(features, repeats=self.mc_samples, axis=0)
            values = tail(repeated, training=self.dropout_mode == "stochastic")
            values = tf.reshape(values, [tf.shape(inputs)[0], self.mc_samples,
                                         self.representation.response_dimension])
            return tf.reduce_mean(values, axis=1), tf.math.reduce_std(values, axis=1)

        return predict

    @classmethod
    def from_checkpoint(cls, checkpoint, representation, **kwargs):
        return cls(load_nacre_forward_model(checkpoint), representation, **kwargs)

    def predict_mc_moments(self, designs):
        tf = require_tensorflow()
        inputs = self.representation.make_inputs(designs)
        means, stds = [], []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start:start + self.batch_size]
            mean, std = self._mc_predict(tf.convert_to_tensor(chunk))
            means.append(mean.numpy())
            stds.append(std.numpy())
        if not means:
            shape = (0, self.representation.response_dimension)
            return np.empty(shape), np.empty(shape)
        return np.concatenate(means).astype(np.float64), np.concatenate(stds).astype(np.float64)

    def predict_mean(self, designs):
        return self.predict_mc_moments(designs)[0]

    def predict_distribution(self, designs):
        mean, std = self.predict_mc_moments(designs)
        covariance = kernel_covariance(std, self.representation.model_strains,
                                       length_scale=self.kernel_length_scale, jitter=self.jitter,
                                       std_scale=self.std_scale)
        return PredictiveDistribution(mean=mean, covariance=covariance)
