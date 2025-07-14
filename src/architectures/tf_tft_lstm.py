from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dropout, LayerNormalization,
    Dense, GlobalAveragePooling1D, Concatenate, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def build_model(config, lookback, n_price_features, n_macro_features):
    H = config.get("horizon", 1)

    # Units
    lstm_units_1 = config.get("units_price_1", 128)
    lstm_units_2 = config.get("units_price_2", 128)
    lstm_units_3 = config.get("units_price_3", 64)
    macro_attn_dim = config.get("macro_attention_dim", 16)
    macro_attn_heads = config.get("macro_attention_heads", 4)

    # Dense fusion units
    dense_1_units = config.get("fused_dense_units_1", 64)
    dense_2_units = config.get("fused_dense_units_2", 32)

    # Dropout
    dropout_price = config.get("dropout_price", 0.0)

    dropout_macro = config.get("dropout_macro", 0.0)
    dropout_fused = config.get("dropout_fused", 0.0)

    # Learning rate and loss
    learning_rate = config.get("learning_rate", 0.001)
    loss_fn = config.get("loss", "huber")

    # Inputs
    price_input = Input(shape=(lookback, n_price_features), name="price_input")
    macro_input = Input(shape=(lookback, n_macro_features), name="macro_input")

    # Price - LSTM stack
    lstm_out = LSTM(lstm_units_1, return_sequences=True)(price_input)
    lstm_out = LSTM(lstm_units_2, return_sequences=True)(lstm_out)
    lstm_out = Dropout(dropout_price)(lstm_out)  # ⬅️ New dropout added here
    lstm_out = LSTM(lstm_units_3, return_sequences=True)(lstm_out)
    lstm_out = LayerNormalization()(lstm_out)

    # Macro - Transformer encoder style
    attn_out = MultiHeadAttention(num_heads=macro_attn_heads, key_dim=macro_attn_dim)(macro_input, macro_input)
    attn_out = Dropout(dropout_macro)(attn_out)
    attn_out = LayerNormalization()(attn_out)
    macro_encoded = Dense(64, activation="gelu")(attn_out)

    # Cross attention: LSTM → attends to macro
    cross_attn = MultiHeadAttention(num_heads=macro_attn_heads, key_dim=macro_attn_dim)(lstm_out, macro_encoded)
    cross_attn = Dropout(dropout_macro)(cross_attn)
    cross_attn = LayerNormalization()(cross_attn)

    # Fusion
    fused = Concatenate()([lstm_out, cross_attn])
    fused = Dense(dense_1_units, activation="gelu")(fused)
    fused = Dropout(dropout_fused)(fused)
    fused = Dense(dense_2_units, activation="gelu")(fused)

    pooled = GlobalAveragePooling1D()(fused)
    output = Dense(H, activation="linear", name="price_prediction")(pooled)

    model = Model(inputs=[price_input, macro_input], outputs=output)

    if loss_fn.lower() == "huber":
        loss = Huber()
    else:
        loss = loss_fn  # e.g. "mse", "mae"

    model.compile(optimizer=Adam(learning_rate), loss=loss, metrics=["mae", "mse"])
    return model
