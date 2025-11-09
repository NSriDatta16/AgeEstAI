# training/2_train_emotion_only.py
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers as L, Model

ROOT   = Path(__file__).resolve().parents[1]
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS_WARM = 4         # freeze backbone
EPOCHS_FT   = 10        # unfreeze backbone
LR_WARM     = 1e-3      # constant LR (float) -> ReduceLROnPlateau works
LR_FT       = 5e-5

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE[0] + 8, IMG_SIZE[1] + 8))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_crop(img, size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    return img

def inverse_freq_weights(indices, n_classes):
    from collections import Counter
    cnt = Counter(indices.tolist() if isinstance(indices, np.ndarray) else indices)
    total = sum(cnt.values())
    return tf.constant([ total / (n_classes * max(1, cnt.get(i, 0))) for i in range(n_classes) ], tf.float32)

def make_emotion_ds(csv_path, batch=BATCH, val_split=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(EMOTIONS)].copy()
    if len(df) == 0:
        raise RuntimeError("emotion.csv empty. Re-run 1_prepare_data.py")

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed, stratify=df["emotion"])
    emo_map = {k:i for i,k in enumerate(EMOTIONS)}
    emo_idx_train = train_df["emotion"].map(emo_map).values
    emo_w_vec = inverse_freq_weights(emo_idx_train, len(EMOTIONS))

    def df_to_ds(frame, augment_flag, use_weights):
        paths  = frame["image_path"].values
        emo_ix = frame["emotion"].map(emo_map).values
        ds = tf.data.Dataset.from_tensor_slices((paths, emo_ix))

        def mapper(p, i):
            img = decode_img(p)
            if augment_flag:
                img = augment(img)
            else:
                img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0], IMG_SIZE[1])
            y = tf.one_hot(i, len(EMOTIONS))
            if use_weights:
                w = tf.gather(emo_w_vec, i)
                return img, y, w
            else:
                return img, y

        if augment_flag:
            ds = ds.shuffle(8192, reshuffle_each_iteration=True)
        ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
        return ds

    return (
        df,
        df_to_ds(train_df, True,  True),
        df_to_ds(val_df,   False, False)
    )

def build_backbone(trainable: bool):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    base.trainable = trainable
    x = L.Dropout(0.25)(base.output)
    return base.input, x

class EmotionNet:
    def __init__(self, train_backbone: bool):
        inp, x = build_backbone(trainable=train_backbone)
        h  = L.Dense(192, activation="relu")(x)
        h  = L.Dropout(0.2)(h)
        out = L.Dense(len(EMOTIONS), activation="softmax")(h)
        self.model = Model(inp, out)

def compile_and_fit(model_obj, train_ds, val_ds, name, epochs, lr):
    model = model_obj.model
    # NOTE: constant LR (float) so ReduceLROnPlateau can adjust it
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    out = MODELS / f"{name}.keras"
    model.save(out)
    print(f"Saved -> {out}")
    return model

def quick_eval(model, val_ds):
    y_true, y_pred = [], []
    for imgs, ys in val_ds.take(80):
        p = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_true.extend(np.argmax(ys.numpy(), axis=1))
    print("\n[Emotion] sample validation report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))
    print(confusion_matrix(y_true, y_pred))

def main():
    emo_csv = PROC / "emotion.csv"
    if not emo_csv.exists():
        print("emotion.csv not found.")
        return

    print("Preparing Emotion dataset...")
    emo_df, emo_train, emo_val = make_emotion_ds(emo_csv)
    print(f"Emotion samples: {len(emo_df)}")

    # warm-up (frozen)
    en = EmotionNet(train_backbone=False)
    m = compile_and_fit(en, emo_train, emo_val, "emotion", EPOCHS_WARM, LR_WARM)
    quick_eval(m, emo_val)

    # fine-tune (unfrozen)
    en_ft = EmotionNet(train_backbone=True)
    mft = compile_and_fit(en_ft, emo_train, emo_val, "emotion_finetuned", EPOCHS_FT, LR_FT)
    quick_eval(mft, emo_val)

if __name__ == "__main__":
    main()
