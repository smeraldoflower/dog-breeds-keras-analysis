import os
import tensorflow as tf
import numpy as np

def main():
    img_height = 180
    img_width = 180

    train_data = tf.keras.utils.image_dataset_from_directory(
        "dogs",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32
    )

    class_names = train_data.class_names
    print(class_names) # ['Basset Hound', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'Great Pyrenees', 'Japanese Chin', 'Shiba Inu', 'Yorkshire Terrier']
    num_classes = len(class_names)

    # recipe for model
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # bake the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # fit the model with training data
    model.fit(train_data, epochs=15)

    # save model
    model.save("model.keras")

main()