import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from typing import Tuple

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TimeSeriesTransformer:
    def __init__(self, 
                 input_shape: Tuple[int, int], 
                 num_classes: int = 3,
                 head_size: int = 256,
                 num_heads: int = 4,
                 ff_dim: int = 4,
                 num_transformer_blocks: int = 4,
                 mlp_units: list = [128],
                 dropout: float = 0.4,
                 learning_rate: float = 1e-4):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Transformer Blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(self.input_shape[1], self.num_heads, self.ff_dim, self.dropout)(x)

        # Global Average Pooling to flatten
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP Head
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.dropout)(x)
            
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"]
        )
        return model

    def summary(self):
        return self.model.summary()
    
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = models.load_model(path, custom_objects={'TransformerBlock': TransformerBlock})
