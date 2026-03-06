import tensorflow as tf
import re

BATCH = 32
EMBEDDING = 128
HIDDEN = 512
SEQ_LEN = 128
NUM_LAYERS = 4

text = open("corpus.txt", "r", encoding="utf-8").read()
text = text.replace("\r\n", "\n").replace("\r", "\n").replace("/", "")
text = text.replace("\t", " ")
text = re.sub(r"[ ]{2,}", " ", text)

text = (text.replace("…", ".")
            .replace("—", "-")
            .replace("–", "-")
            .replace("“", '"').replace("”", '"')
            .replace("’", "'"))

vocab = sorted(set(text))
str2id = {c: i for i, c in enumerate(vocab)}
id2str = {i: c for c, i in str2id.items()}

ids = [str2id[c] for c in text]
ids_tf = tf.constant(ids, dtype=tf.int32)

def split(part):
    x = part[:-1]
    y = part[1:]
    return x, y

CHARNUM = len(vocab)

def make_stateless_ds():
    ds = tf.data.Dataset.from_tensor_slices(ids_tf)
    ds = ds.batch(SEQ_LEN + 1, drop_remainder=True)
    ds = ds.map(split, num_parallel_calls=tf.data.AUTOTUNE)
    NUMPARTS = (len(ids) - 1) // (SEQ_LEN + 1)
    ds = ds.shuffle(NUMPARTS)
    ds = ds.batch(BATCH, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE), NUMPARTS // BATCH

def build_model_stateless(layers):
    inp = tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(CHARNUM, EMBEDDING)(inp)
    for _ in range(layers):
        x = tf.keras.layers.LSTM(
            HIDDEN,
            return_sequences=True,
            activation="tanh"
        )(x)
    preds = tf.keras.layers.Dense(CHARNUM)(x)
    return tf.keras.Model(inp, preds)

def build_model_stateful(num_layers, batch_size, seq_len):
    inp = tf.keras.Input(batch_shape=(batch_size, seq_len), dtype=tf.int32)
    x = tf.keras.layers.Embedding(CHARNUM, EMBEDDING)(inp)
    for _ in range(num_layers):
        x = tf.keras.layers.LSTM(
            HIDDEN, return_sequences=True,
            activation="tanh",
            stateful=True
        )(x)
    preds = tf.keras.layers.Dense(CHARNUM)(x)
    return tf.keras.Model(inp, preds)

def get_next_temperature(preds, temp):
    if temp == 0:
        return int(tf.argmax(preds).numpy())
    scaled = preds / temp
    nxt = tf.random.categorical(tf.expand_dims(scaled, 0), 1)[0, 0]
    return int(nxt.numpy())

def reset_states(model):
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()

def generate_stateful(mdl_gen, start_text, length, temp):
    reset_states(mdl_gen)
    prompt_ids = [str2id.get(c, 0) for c in start_text] or [0]
    out = list(start_text)
    for pid in prompt_ids[:-1]:
        _ = mdl_gen(tf.constant([[pid]], dtype=tf.int32), training=False)
    last = prompt_ids[-1]
    for _ in range(length):
        logits = mdl_gen(tf.constant([[last]], dtype=tf.int32), training=False)[0, -1, :]
        nxt = get_next_temperature(logits, temp)
        out.append(id2str[nxt])
        last = nxt
    return "".join(out)

ds, steps_per_epoch = make_stateless_ds()
mdl = build_model_stateless(NUM_LAYERS)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

total_steps = max(1, steps_per_epoch * 10)
lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=total_steps, alpha=0.1)

mdl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
    loss=loss
)

mdl.fit(ds, epochs=10)

mdl_gen = build_model_stateful(NUM_LAYERS, 1, 1)
mdl_gen.set_weights(mdl.get_weights())

start1 = "KSIĄDZ\n\n"
start2 = "GUSTAW\n\n"

print(generate_stateful(mdl_gen, start1, 600, 0.8))
print(generate_stateful(mdl_gen, start2, 600, 0.8))
print(generate_stateful(mdl_gen, start1, 600, 1.0))