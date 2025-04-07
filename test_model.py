import os
import numpy as np
import tensorflow as tf 


def test_model(img_height, img_width, class_names, model):
    print("********* Predicting dog breed for test images *********")
    for filename in os.listdir("test_dogs"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                print(filename)
                img = tf.keras.utils.load_img("test_dogs/" + filename, 
                                        target_size=(img_height, img_width))

                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # create a batch

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
                )

def main():
    img_height = 180
    img_width = 180
    class_names = ['Basset Hound', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'Great Pyrenees', 'Japanese Chin', 'Shiba Inu', 'Yorkshire Terrier']

    model = tf.keras.models.load_model('model.keras')

    # test the model
    test_model(img_height, img_width, class_names, model)

main()