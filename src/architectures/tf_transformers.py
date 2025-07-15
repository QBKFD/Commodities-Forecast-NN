from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Concatenate,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

def build_model(config, lookback, n_price_features, n_macro_features):

    lr = config.get("learning_rate", 0.01)
    loss_fn = config.get("loss", "mse")
    dropout_price = config.get("dropout_price", 0.0)
    dropout_macro = config.get("dropout_macro", 0.0)
    dropout_cross = config.get("dropout_cross", 0.0)
    dropout_fused = config.get("dropout_fused", 0.001)

    price_units = config.get("price_dense_units", 64)
    macro_units = config.get("macro_dense_units", 64)
    fused_units_1 = config.get("fused_dense_units_1", 32)
    fused_units_2 = config.get("fused_dense_units_2", 32)

    attn_heads = config.get("attention_heads", 4)
    attn_key_dim = config.get("attention_key_dim", 16)

    horizon = config.get("horizon", 1)

    price_input = Input(shape=(lookback, n_price_features), name='price_input')
    macro_input = Input(shape=(lookback, n_macro_features), name='macro_input')

    price_attn = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(price_input, price_input)
    price_attn = Dropout(dropout_price)(price_attn)
    price_attn = LayerNormalization()(price_attn)
    price_attn = Dense(price_units, activation='relu')(price_attn)

    macro_attn = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(macro_input, macro_input)
    macro_attn = Dropout(dropout_macro)(macro_attn)
    macro_attn = LayerNormalization()(macro_attn)
    macro_attn = Dense(macro_units, activation='relu')(macro_attn)

    cross_attn = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(price_attn, macro_attn)
    cross_attn = Dropout(dropout_cross)(cross_attn)
    cross_attn = LayerNormalization()(cross_attn)

    
    fused = Concatenate()([price_attn, cross_attn])
    fused = Dense(fused_units_1, activation='relu')(fused)
    fused = Dropout(dropout_fused)(fused)
    fused = Dense(fused_units_2, activation='relu')(fused)

    pooled = GlobalAveragePooling1D()(fused)
    output = Dense(horizon, activation='linear', name='price_prediction')(pooled)

    model = Model(inputs=[price_input, macro_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=["mae", "mse"])

    return model
