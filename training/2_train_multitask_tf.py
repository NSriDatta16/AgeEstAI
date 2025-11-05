# training/2_train_multitask_tf.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers as L, Model

# -----------------------
# Paths & constants
# -----------------------
ROOT   = Path(__file__).resolve().parents[1]
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 10  # start small; you can raise later (e.g., 20-30)

AGE_BINS = ["0-12","13-19","20-29","30-39","40-49","50-64","65+"]
GENDERS  = ["female","male"]
EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# -----------------------
# Utilities
# -----------------------
def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

def build_backbone():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False  # warm-up; you can unfreeze later
    x = L.Dropout(0.3)(base.output)
    return base.input, x

# -----------------------
# Age + Gender dataset
# -----------------------
def make_age_gender_ds(csv_path, batch=BATCH, val_split=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df["age_bin"].isin(AGE_BINS) & df["gender"].isin(GENDERS)].copy()

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed,
        stratify=df[["age_bin","gender"]]
    )

    age_map = {k:i for i,k in enumerate(AGE_BINS)}
    gen_map = {k:i for i,k in enumerate(GENDERS)}

    def df_to_ds(frame, augment_flag):
        paths   = frame["image_path"].values
        age_oh  = tf.one_hot(frame["age_bin"].map(age_map).values, len(AGE_BINS))
        gen_oh  = tf.one_hot(frame["gender"].map(gen_map).values, len(GENDERS))
        ds = tf.data.Dataset.from_tensor_slices((paths, (age_oh, gen_oh)))

        def mapper(p, labels):
            img = decode_img(p)
            if augment_flag:
                img = augment(img)
            return img, {"age": labels[0], "gender": labels[1]}

        ds = ds.shuffle(4096, reshuffle_each_iteration=True) if augment_flag else ds
        ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
        return ds

    return df, df_to_ds(train_df, True), df_to_ds(val_df, False)

class AgeGenderNet:
    def __init__(self):
        inp, x = build_backbone()

        age = L.Dense(128, activation="relu")(x)
        age = L.Dense(len(AGE_BINS), activation="softmax", name="age")(age)

        gen = L.Dense(64, activation="relu")(x)
        gen = L.Dense(len(GENDERS), activation="softmax", name="gender")(gen)

        self.model = Model(inp, [age, gen])
        self.loss  = {"age":"categorical_crossentropy","gender":"categorical_crossentropy"}
        self.metrics = {"age":"accuracy","gender":"accuracy"}

# -----------------------
# Emotion dataset
# -----------------------
def make_emotion_ds(csv_path, batch=BATCH, val_split=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(EMOTIONS)].copy()

    if len(df) == 0:
        raise RuntimeError("emotion.csv is empty. Re-run 1_prepare_data.py and confirm RAF-DB was parsed.")

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["emotion"]
    )

    emo_map = {k:i for i,k in enumerate(EMOTIONS)}

    def df_to_ds(frame, augment_flag):
        paths  = frame["image_path"].values
        emo_oh = tf.one_hot(frame["emotion"].map(emo_map).values, len(EMOTIONS))
        ds = tf.data.Dataset.from_tensor_slices((paths, emo_oh))

        def mapper(p, y):
            img = decode_img(p)
            if augment_flag:
                img = augment(img)
            return img, {"emotion": y}

        ds = ds.shuffle(4096, reshuffle_each_iteration=True) if augment_flag else ds
        ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
        return ds

    return df, df_to_ds(train_df, True), df_to_ds(val_df, False)

class EmotionNet:
    def __init__(self):
        inp, x = build_backbone()
        emo = L.Dense(128, activation="relu")(x)
        emo = L.Dense(len(EMOTIONS), activation="softmax", name="emotion")(emo)
        self.model = Model(inp, emo)
        self.loss  = {"emotion":"categorical_crossentropy"}
        self.metrics = {"emotion":"accuracy"}

# -----------------------
# Train helper
# -----------------------
def train_compile_fit(model_obj, train_ds, val_ds, name, epochs=EPOCHS):
    model = model_obj.model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=model_obj.loss,
        metrics=model_obj.metrics,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    out_path = MODELS / f"{name}.keras"
    model.save(out_path)
    print(f"Saved -> {out_path}")

# -----------------------
# Main
# -----------------------
def main():
    # Age+Gender
    ag_csv = PROC / "age_gender.csv"
    if ag_csv.exists():
        print("Preparing Age+Gender dataset...")
        ag_df, ag_train, ag_val = make_age_gender_ds(ag_csv)
        print(f"Age+Gender samples: {len(ag_df)}  (train≈{len(ag_df)*0.9:.0f} / val≈{len(ag_df)*0.1:.0f})")
        ag = AgeGenderNet()
        train_compile_fit(ag, ag_train, ag_val, name="age_gender", epochs=EPOCHS)
    else:
        print("age_gender.csv not found — skipping.")

    # Emotion
    emo_csv = PROC / "emotion.csv"
    if emo_csv.exists():
        print("Preparing Emotion dataset...")
        emo_df, emo_train, emo_val = make_emotion_ds(emo_csv)
        print(f"Emotion samples: {len(emo_df)}  (train≈{len(emo_df)*0.9:.0f} / val≈{len(emo_df)*0.1:.0f})")
        en = EmotionNet()
        train_compile_fit(en, emo_train, emo_val, name="emotion", epochs=EPOCHS)
    else:
        print("emotion.csv not found — skipping.")

if __name__ == "__main__":
    main()
