from __future__ import annotations

from collections.abc import Callable

from .config import ModelConfig
from .features import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, MODEL_FEATURES


def build_model_factory(
    cardinalities: dict[str, int],
    config: ModelConfig,
    seed: int,
    model_name: str = "FederatedDeepFM",
) -> Callable:
    """Return the zero-argument Keras builder required by SecretFlow FLModel."""

    def model_builder():
        import numpy as np
        import tensorflow as tf

        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        regularizer = tf.keras.regularizers.l2(config.l2_regularization)
        inputs = {
            name: tf.keras.layers.Input(
                shape=(1,),
                name=name,
                dtype=tf.int32 if name in CATEGORICAL_FEATURES else tf.float32,
            )
            for name in MODEL_FEATURES
        }

        embeddings = []
        linear_terms = []
        for name in CATEGORICAL_FEATURES:
            embedding = tf.keras.layers.Embedding(
                cardinalities[name],
                config.embedding_dim,
                embeddings_regularizer=regularizer,
                name=f"{name}_embedding",
            )(inputs[name])
            embeddings.append(tf.keras.layers.Flatten()(embedding))
            linear_terms.append(
                tf.keras.layers.Flatten()(
                    tf.keras.layers.Embedding(
                        cardinalities[name],
                        1,
                        embeddings_initializer="zeros",
                        embeddings_regularizer=regularizer,
                        name=f"{name}_linear",
                    )(inputs[name])
                )
            )

        numeric = tf.keras.layers.Concatenate(name="numeric")(
            [inputs[name] for name in CONTINUOUS_FEATURES]
        )
        linear = tf.keras.layers.Add(name="linear_categorical")(linear_terms)
        linear = tf.keras.layers.Add(name="linear_all")(
            [
                linear,
                tf.keras.layers.Dense(
                    1, kernel_regularizer=regularizer, name="linear_numeric"
                )(numeric),
            ]
        )
        pairwise = [
            tf.keras.layers.Dot(axes=1, name=f"fm_{left}_{right}")(
                [embeddings[left], embeddings[right]]
            )
            for left in range(len(embeddings))
            for right in range(left + 1, len(embeddings))
        ]
        fm = tf.keras.layers.Add(name="fm_interactions")(pairwise)

        deep = tf.keras.layers.Concatenate(name="deep_input")([*embeddings, numeric])
        for index, units in enumerate(config.hidden_units, start=1):
            deep = tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizer,
                name=f"deep_{index}",
            )(deep)
            deep = tf.keras.layers.Dropout(
                config.dropout, name=f"dropout_{index}"
            )(deep)
        deep = tf.keras.layers.Dense(
            1, kernel_regularizer=regularizer, name="deep_logit"
        )(deep)
        logit = tf.keras.layers.Add(name="deepfm_logit")([linear, fm, deep])
        probability = tf.keras.layers.Activation(
            "sigmoid", name="preference_probability"
        )(logit)

        def hybrid_bce_focal(y_true, y_pred):
            y_true_cast = tf.cast(y_true, y_pred.dtype)
            epsilon = tf.keras.backend.epsilon()
            predicted = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -(
                y_true_cast * tf.math.log(predicted)
                + (1.0 - y_true_cast) * tf.math.log(1.0 - predicted)
            )
            probability_true = (
                y_true_cast * predicted
                + (1.0 - y_true_cast) * (1.0 - predicted)
            )
            focal = tf.pow(1.0 - probability_true, config.focal_gamma) * cross_entropy
            return config.bce_weight * cross_entropy + (1.0 - config.bce_weight) * focal

        model = tf.keras.Model(
            [inputs[name] for name in MODEL_FEATURES], probability, name=model_name
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=hybrid_bce_focal,
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    return model_builder

