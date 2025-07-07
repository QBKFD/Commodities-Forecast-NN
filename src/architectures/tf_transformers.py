from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Concatenate,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

def build_model(config, lookback, n_price_features, n_macro_features):
    lr = config.get("learning_rate", 0.01)
    dropout = config.get("dropout", 0.001)
    dense_units_1 = config.get("fused_dense_units_1", 32)
    dense_units_2 = config.get("fused_dense_units_2", 32)
    loss_fn = config.get("loss", "mse")
    horizon = config.get("horizon", 1)

    price_input = Input(shape=(lookback, n_price_features), name='price_input')
    macro_input = Input(shape=(lookback, n_macro_features), name='macro_input')

    # Transformer for price
    price_attn = MultiHeadAttention(num_heads=4, key_dim=16)(price_input, price_input)
    price_attn = Dropout(dropout)(price_attn)
    price_attn = LayerNormalization()(price_attn)
    price_attn = Dense(64, activation='relu')(price_attn)

    # Transformer for macro
    macro_attn = MultiHeadAttention(num_heads=4, key_dim=16)(macro_input, macro_input)
    macro_attn = Dropout(dropout)(macro_attn)
    macro_attn = LayerNormalization()(macro_attn)
    macro_attn = Dense(64, activation='relu')(macro_attn)

    # Cross Attention
    cross_attn = MultiHeadAttention(num_heads=4, key_dim=16)(price_attn, macro_attn)
    cross_attn = Dropout(dropout)(cross_attn)
    cross_attn = LayerNormalization()(cross_attn)

    # Fusion
    fused = Concatenate()([price_attn, cross_attn])
    fused = Dense(dense_units_1, activation='relu')(fused)
    fused = Dropout(dropout)(fused)
    fused = Dense(dense_units_2, activation='relu')(fused)

    pooled = GlobalAveragePooling1D()(fused)
    output = Dense(horizon, activation='linear', name='price_prediction')(pooled)

    model = Model(inputs=[price_input, macro_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=['mae', 'mse'])

    return model
