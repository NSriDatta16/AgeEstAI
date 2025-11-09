# training/2_train_multitask_tf.py
from pathlib import Path
from collections import Counter
import math
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

# -----------------------
# Config
# -----------------------
IMG_SIZE = (224, 224)          # slightly smaller → better FPS + training speed
BATCH    = 32
EPOCHS_BASE = 6                # warmup (frozen backbone)
EPOCHS_FT   = 12               # fine-tune (unfrozen)
LR_BASE     = 1e-3
LR_FT       = 5e-5

AGE_BINS = ["0-12","13-19","20-29","30-39","40-49","50-64","65+"]
GENDERS  = ["female","male"]
EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# centers used for expected-age estimation (years)
AGE_BIN_CENTERS = np.array([6, 16, 25, 35, 45, 57, 70], dtype=np.float32)

# -----------------------
# Image utils / aug
# -----------------------
def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE[0] + 8, IMG_SIZE[1] + 8))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def rand_color_jitter(img):
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    return img

def cutout(img, max_frac=0.25):
    h, w = IMG_SIZE
    fh = tf.random.uniform([], 0, int(h * max_frac), dtype=tf.int32)
    fw = tf.random.uniform([], 0, int(w * max_frac), dtype=tf.int32)
    y0 = tf.random.uniform([], 0, h - fh, dtype=tf.int32)
    x0 = tf.random.uniform([], 0, w - fw, dtype=tf.int32)
    mask = tf.ones((fh, fw, 3), dtype=img.dtype) * 0.0
    pad_top    = y0
    pad_bottom = h - fh - y0
    pad_left   = x0
    pad_right  = w - fw - x0
    mask = tf.pad(mask, [[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], constant_values=1.0)
    return img * mask

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = rand_color_jitter(img)
    # random crop around center to introduce framing jitter
    img = tf.image.random_crop(img, size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    if tf.random.uniform([]) < 0.4:
        img = cutout(img, max_frac=0.22)
    return img

# -----------------------
# Backbone
# -----------------------
def build_backbone(trainable: bool):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    base.trainable = trainable
    x = L.Dropout(0.35)(base.output)
    return base.input, x

# -----------------------
# Helpers
# -----------------------
def inverse_freq_weights(indices, n_classes):
    cnt = Counter(indices.tolist() if isinstance(indices, np.ndarray) else indices)
    total = sum(cnt.values())
    return tf.constant([ total / (n_classes * max(1, cnt.get(i, 0))) for i in range(n_classes) ], tf.float32)

def gaussian_soft_one_hot(idx, n_classes, sigma=0.85):
    # Create soft label peaking at idx with Gaussian falloff over class index
    xs = tf.range(n_classes, dtype=tf.float32)
    dist2 = (xs - tf.cast(idx, tf.float32))**2
    logits = tf.exp(-dist2 / (2.0 * (sigma**2)))
    probs = logits / tf.reduce_sum(logits)
    return probs

# -----------------------
# Age + Gender dataset (with SOFT age labels)
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

    age_idx_train = train_df["age_bin"].map(age_map).values
    gen_idx_train = train_df["gender"].map(gen_map).values
    age_w_vec = inverse_freq_weights(age_idx_train, len(AGE_BINS))
    gen_w_vec = inverse_freq_weights(gen_idx_train, len(GENDERS))

    def df_to_ds(frame, augment_flag, use_weights):
        paths   = frame["image_path"].values
        age_idx = frame["age_bin"].map(age_map).values
        gen_idx = frame["gender"].map(gen_map).values

        ds = tf.data.Dataset.from_tensor_slices((paths, age_idx, gen_idx))

        def mapper(p, a_idx, g_idx):
            img = decode_img(p)
            if augment_flag:
                img = augment(img)
            else:
                img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0], IMG_SIZE[1])

            # SOFT age one-hot
            y_age = gaussian_soft_one_hot(a_idx, len(AGE_BINS), sigma=0.85)
            # Gender standard one-hot
            y_gen = tf.one_hot(g_idx, len(GENDERS))

            if use_weights:
                w_age = tf.gather(age_w_vec, a_idx)
                w_gen = tf.gather(gen_w_vec, g_idx)
                sw = (w_age, w_gen)    # tuple aligned with outputs [age, gender]
                return img, (y_age, y_gen), sw
            else:
                return img, (y_age, y_gen)

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

class AgeGenderNet:
    def __init__(self, train_backbone: bool):
        inp, x = build_backbone(trainable=train_backbone)
        age = L.Dense(192, activation="relu")(x)
        age = L.Dropout(0.2)(age)
        age = L.Dense(len(AGE_BINS), activation="softmax")(age)

        gen = L.Dense(128, activation="relu")(x)
        gen = L.Dropout(0.15)(gen)
        gen = L.Dense(len(GENDERS), activation="softmax")(gen)

        self.model = Model(inp, [age, gen])
        # list-style to match tuple labels
        self.loss     = ["categorical_crossentropy", "categorical_crossentropy"]
        self.metrics  = ["accuracy", "accuracy"]

# -----------------------
# Emotion dataset / model (unchanged except compile metrics list)
# -----------------------
def make_emotion_ds(csv_path, batch=BATCH, val_split=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(EMOTIONS)].copy()
    if len(df) == 0:
        raise RuntimeError("emotion.csv is empty. Re-run 1_prepare_data.py and ensure RAF-DB parsed.")

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

class EmotionNet:
    def __init__(self, train_backbone: bool):
        inp, x = build_backbone(trainable=train_backbone)
        emo = L.Dense(192, activation="relu")(x)
        emo = L.Dropout(0.2)(emo)
        emo = L.Dense(len(EMOTIONS), activation="softmax")(emo)
        self.model = Model(inp, emo)
        self.loss     = "categorical_crossentropy"
        self.metrics  = ["accuracy"]

# -----------------------
# Train / Eval
# -----------------------
def compile_and_fit(model_obj, train_ds, val_ds, name, epochs, lr):
    model = model_obj.model
    # Cosine decay improves late fine-tuning
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr, decay_steps=epochs*1000))
    model.compile(optimizer=opt, loss=model_obj.loss, metrics=model_obj.metrics)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    out = MODELS / f"{name}.keras"
    model.save(out)
    print(f"Saved -> {out}")
    return model

def quick_eval_age_gender(model, val_ds):
    y_age_true, y_age_pred = [], []
    y_gen_true, y_gen_pred = [], []
    exp_ages = []
    for imgs, ys in val_ds.take(200):
        p_age, p_gen = model.predict(imgs, verbose=0)
        y_age_pred.extend(np.argmax(p_age, axis=1))
        y_gen_pred.extend(np.argmax(p_gen, axis=1))
        y_age_true.extend(np.argmax(ys[0].numpy(), axis=1))
        y_gen_true.extend(np.argmax(ys[1].numpy(), axis=1))
        # expected age in years from softmax probs
        exp_ages.extend((p_age @ AGE_BIN_CENTERS).tolist())
    print("\n[Age] sample validation report:")
    print(classification_report(y_age_true, y_age_pred, target_names=AGE_BINS, digits=4))
    print(confusion_matrix(y_age_true, y_age_pred))
    print("[Age] sample expected age (years) mean±std:", np.mean(exp_ages), "±", np.std(exp_ages))
    print("\n[Gender] sample validation report:")
    print(classification_report(y_gen_true, y_gen_pred, target_names=GENDERS, digits=4))
    print(confusion_matrix(y_gen_true, y_gen_pred))

def quick_eval_emotion(model, val_ds):
    y_true, y_pred = [], []
    for imgs, ys in val_ds.take(50):
        p = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_true.extend(np.argmax(ys.numpy(), axis=1))
    print("\n[Emotion] sample validation report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))
    print(confusion_matrix(y_true, y_pred))

# -----------------------
# Main
# -----------------------
def main():
    ag_csv = PROC / "age_gender.csv"
    if ag_csv.exists():
        print("Preparing Age+Gender dataset...")
        ag_df, ag_train, ag_val = make_age_gender_ds(ag_csv)
        print(f"Age+Gender samples: {len(ag_df)}")

        # Warm-up (frozen backbone)
        base = AgeGenderNet(train_backbone=False)
        m = compile_and_fit(base, ag_train, ag_val, "age_gender", EPOCHS_BASE, LR_BASE)
        quick_eval_age_gender(m, ag_val)

        # Fine-tune (unfrozen)
        ft = AgeGenderNet(train_backbone=True)
        mft = compile_and_fit(ft, ag_train, ag_val, "age_gender_finetuned", EPOCHS_FT, LR_FT)
        quick_eval_age_gender(mft, ag_val)
    else:
        print("age_gender.csv not found — skipping Age+Gender.")

    emo_csv = PROC / "emotion.csv"
    if emo_csv.exists():
        print("Preparing Emotion dataset...")
        emo_df, emo_train, emo_val = make_emotion_ds(emo_csv)
        print(f"Emotion samples: {len(emo_df)}")
        en = EmotionNet(train_backbone=False)
        me = compile_and_fit(en, emo_train, emo_val, "emotion", EPOCHS_BASE, LR_BASE)
        quick_eval_emotion(me, emo_val)
        en_ft = EmotionNet(train_backbone=True)
        me_ft = compile_and_fit(en_ft, emo_train, emo_val, "emotion_finetuned", EPOCHS_FT, LR_FT)
        quick_eval_emotion(me_ft, emo_val)
    else:
        print("emotion.csv not found — skipping Emotion.")

if __name__ == "__main__":
    main()
