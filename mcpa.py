import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224  # Input image size (standard for ViT)
PATCH_SIZE = 16  # Patch size for ViT
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
EMBED_DIM = 256  # Embedding dimension for patches
NUM_HEADS = 8  # Number of attention heads
FFN_DIM = 512  # Feed-forward network dimension
NUM_LAYERS = 6  # Number of transformer layers
NUM_CLASSES = 10  # CIFAR-10 classes
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4

# Load and preprocess CIFAR-10 dataset
def load_and_preprocess_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10', split=['train', 'test'], as_supervised=True, with_info=True
    )

    def preprocess(image, label):
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def apply_augmentation(image, label):
        image = data_augmentation(image, training=True)
        return image, label

    ds_train = ds_train.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

# MixUp augmentation
def mixup(images, labels, alpha=0.4):
    batch_size = tf.shape(images)[0]
    lam = tf.random.stateless_uniform([], seed=(1, 2), minval=0, maxval=alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    images_mixed = lam * images + (1 - lam) * tf.gather(images, indices)
    labels_mixed = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return images_mixed, labels_mixed

# Patch extraction layer
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch embedding layer
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = self.add_weight(
            "pos_embed", shape=(1, num_patches + 1, embed_dim), initializer="zeros"
        )
        self.cls_token = self.add_weight(
            "cls_token", shape=(1, 1, embed_dim), initializer="zeros"
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        patches = self.proj(patches)
        x = tf.concat([cls_tokens, patches], axis=1)
        x = x + self.pos_embed
        return x

# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ffn_dim, activation='gelu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.attn(x, x)
        attn_output = self.dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x

# Vision Transformer model
def build_vit():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    patches = PatchExtractor(PATCH_SIZE)(inputs)
    x = PatchEmbedding(NUM_PATCHES, EMBED_DIM)(patches)

    for _ in range(NUM_LAYERS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FFN_DIM)(x, training=True)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]  # Take CLS token
    x = layers.Dense(NUM_CLASSES, kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = layers.Activation('softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Training and evaluation
ds_train, ds_test, ds_info = load_and_preprocess_data()
model = build_vit()

# Compile model with label smoothing
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5
)

# Train model
history = model.fit(
    ds_train.map(mixup, num_parallel_calls=tf.data.AUTOTUNE),
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(ds_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Visualize training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()