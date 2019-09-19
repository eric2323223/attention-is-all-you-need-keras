import numpy as np
from keras_transformer import get_model
from keras.utils.vis_utils import plot_model

# Build a small toy token dictionary
tokens = 'all work and no play makes jack a dull boy'.split(' ')
token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}
for token in tokens:
    if token not in token_dict:
        token_dict[token] = len(token_dict)

# Generate toy data
encoder_inputs_no_padding = []
encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
for i in range(1, len(tokens) - 1):
    encode_tokens, decode_tokens = tokens[:i], tokens[i:]
    encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(encode_tokens))
    output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
    decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
    encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
    decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
    output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
    encoder_inputs_no_padding.append(encode_tokens[:i + 2])
    encoder_inputs.append(encode_tokens)
    decoder_inputs.append(decode_tokens)
    decoder_outputs.append(output_tokens)

print(encoder_inputs)

# Build the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=30,
    encoder_num=3,
    decoder_num=2,
    head_num=3,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((13, 30)),
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Train the model
# model.fit(
#     x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
#     y=np.asarray(decoder_outputs * 1000),
#     epochs=5,
# )


# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# Decoder-Input (InputLayer)      (None, None)         0
# __________________________________________________________________________________________________
# Encoder-Input (InputLayer)      (None, None)         0
# __________________________________________________________________________________________________
# Token-Embedding (EmbeddingRet)  [(None, None, 30), ( 390         Encoder-Input[0][0]
#                                                                  Decoder-Input[0][0]
# __________________________________________________________________________________________________
# Encoder-Embedding (TrigPosEmbed (None, None, 30)     0           Token-Embedding[0][0]
# __________________________________________________________________________________________________
# Encoder-1-MultiHeadSelfAttentio (None, None, 30)     3720        Encoder-Embedding[0][0]
# __________________________________________________________________________________________________
# Encoder-1-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-1-MultiHeadSelfAttention[
# __________________________________________________________________________________________________
# Encoder-1-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-Embedding[0][0]
#                                                                  Encoder-1-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-1-MultiHeadSelfAttentio (None, None, 30)     60          Encoder-1-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-1-FeedForward (FeedForw (None, None, 30)     7350        Encoder-1-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-1-FeedForward-Dropout ( (None, None, 30)     0           Encoder-1-FeedForward[0][0]
# __________________________________________________________________________________________________
# Encoder-1-FeedForward-Add (Add) (None, None, 30)     0           Encoder-1-MultiHeadSelfAttention-
#                                                                  Encoder-1-FeedForward-Dropout[0][
# __________________________________________________________________________________________________
# Encoder-1-FeedForward-Norm (Lay (None, None, 30)     60          Encoder-1-FeedForward-Add[0][0]
# __________________________________________________________________________________________________
# Encoder-2-MultiHeadSelfAttentio (None, None, 30)     3720        Encoder-1-FeedForward-Norm[0][0]
# __________________________________________________________________________________________________
# Encoder-2-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-2-MultiHeadSelfAttention[
# __________________________________________________________________________________________________
# Encoder-2-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-1-FeedForward-Norm[0][0]
#                                                                  Encoder-2-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-2-MultiHeadSelfAttentio (None, None, 30)     60          Encoder-2-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-2-FeedForward (FeedForw (None, None, 30)     7350        Encoder-2-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-2-FeedForward-Dropout ( (None, None, 30)     0           Encoder-2-FeedForward[0][0]
# __________________________________________________________________________________________________
# Encoder-2-FeedForward-Add (Add) (None, None, 30)     0           Encoder-2-MultiHeadSelfAttention-
#                                                                  Encoder-2-FeedForward-Dropout[0][
# __________________________________________________________________________________________________
# Encoder-2-FeedForward-Norm (Lay (None, None, 30)     60          Encoder-2-FeedForward-Add[0][0]
# __________________________________________________________________________________________________
# Encoder-3-MultiHeadSelfAttentio (None, None, 30)     3720        Encoder-2-FeedForward-Norm[0][0]
# __________________________________________________________________________________________________
# Encoder-3-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-3-MultiHeadSelfAttention[
# __________________________________________________________________________________________________
# Encoder-3-MultiHeadSelfAttentio (None, None, 30)     0           Encoder-2-FeedForward-Norm[0][0]
#                                                                  Encoder-3-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Decoder-Embedding (TrigPosEmbed (None, None, 30)     0           Token-Embedding[1][0]
# __________________________________________________________________________________________________
# Encoder-3-MultiHeadSelfAttentio (None, None, 30)     60          Encoder-3-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadSelfAttentio (None, None, 30)     3720        Decoder-Embedding[0][0]
# __________________________________________________________________________________________________
# Encoder-3-FeedForward (FeedForw (None, None, 30)     7350        Encoder-3-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadSelfAttentio (None, None, 30)     0           Decoder-1-MultiHeadSelfAttention[
# __________________________________________________________________________________________________
# Encoder-3-FeedForward-Dropout ( (None, None, 30)     0           Encoder-3-FeedForward[0][0]
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadSelfAttentio (None, None, 30)     0           Decoder-Embedding[0][0]
#                                                                  Decoder-1-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-3-FeedForward-Add (Add) (None, None, 30)     0           Encoder-3-MultiHeadSelfAttention-
#                                                                  Encoder-3-FeedForward-Dropout[0][
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadSelfAttentio (None, None, 30)     60          Decoder-1-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Encoder-3-FeedForward-Norm (Lay (None, None, 30)     60          Encoder-3-FeedForward-Add[0][0]
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadQueryAttenti (None, None, 30)     3720        Decoder-1-MultiHeadSelfAttention-
#                                                                  Encoder-3-FeedForward-Norm[0][0]
#                                                                  Encoder-3-FeedForward-Norm[0][0]
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadQueryAttenti (None, None, 30)     0           Decoder-1-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadQueryAttenti (None, None, 30)     0           Decoder-1-MultiHeadSelfAttention-
#                                                                  Decoder-1-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-1-MultiHeadQueryAttenti (None, None, 30)     60          Decoder-1-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-1-FeedForward (FeedForw (None, None, 30)     7350        Decoder-1-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-1-FeedForward-Dropout ( (None, None, 30)     0           Decoder-1-FeedForward[0][0]
# __________________________________________________________________________________________________
# Decoder-1-FeedForward-Add (Add) (None, None, 30)     0           Decoder-1-MultiHeadQueryAttention
#                                                                  Decoder-1-FeedForward-Dropout[0][
# __________________________________________________________________________________________________
# Decoder-1-FeedForward-Norm (Lay (None, None, 30)     60          Decoder-1-FeedForward-Add[0][0]
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadSelfAttentio (None, None, 30)     3720        Decoder-1-FeedForward-Norm[0][0]
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadSelfAttentio (None, None, 30)     0           Decoder-2-MultiHeadSelfAttention[
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadSelfAttentio (None, None, 30)     0           Decoder-1-FeedForward-Norm[0][0]
#                                                                  Decoder-2-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadSelfAttentio (None, None, 30)     60          Decoder-2-MultiHeadSelfAttention-
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadQueryAttenti (None, None, 30)     3720        Decoder-2-MultiHeadSelfAttention-
#                                                                  Encoder-3-FeedForward-Norm[0][0]
#                                                                  Encoder-3-FeedForward-Norm[0][0]
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadQueryAttenti (None, None, 30)     0           Decoder-2-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadQueryAttenti (None, None, 30)     0           Decoder-2-MultiHeadSelfAttention-
#                                                                  Decoder-2-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-2-MultiHeadQueryAttenti (None, None, 30)     60          Decoder-2-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-2-FeedForward (FeedForw (None, None, 30)     7350        Decoder-2-MultiHeadQueryAttention
# __________________________________________________________________________________________________
# Decoder-2-FeedForward-Dropout ( (None, None, 30)     0           Decoder-2-FeedForward[0][0]
# __________________________________________________________________________________________________
# Decoder-2-FeedForward-Add (Add) (None, None, 30)     0           Decoder-2-MultiHeadQueryAttention
#                                                                  Decoder-2-FeedForward-Dropout[0][
# __________________________________________________________________________________________________
# Decoder-2-FeedForward-Norm (Lay (None, None, 30)     60          Decoder-2-FeedForward-Add[0][0]
# __________________________________________________________________________________________________
# Output (EmbeddingSim)           (None, None, 13)     13          Decoder-2-FeedForward-Norm[0][0]
#                                                                  Token-Embedding[1][1]
# ==================================================================================================
# Total params: 63,913
# Trainable params: 63,523
# Non-trainable params: 390
# __________________________________________________________________________________________________