import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import keras_preprocessing
import keras_cv
from keras_cv import bounding_box
import time
import os
import sys

print('Tensorflow version - ',tf.__version__)
print('Keras_cv version - ', keras_cv.__version__)

checkpoint_path = "training/weights.{epoch:02d}-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
data_directory = 'H:/tensorflow_datasets/voc'
batch_size=64
epoch_count=50

def main():
    args = sys.argv[0:]

    if len(args) == 1:
        print(' ----- checkpoints directory ', checkpoint_path)
    else:
        if len(args) == 3:
            batch_size = int(args[2])
        checkpoint_path = args[1] + "/weights.{epoch:02d}-{accuracy:.3f}-{loss:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
        print(' ----- checkpoints directory ', args[1])
        print(' -- batch size: ', batch_size)
        mobilenet = loadModel()
        run(checkpoint_path, mobilenet)

def loadModel():
    # use MobileNet from Keras applications
    model = keras_cv.models.RetinaNet(classes=20,bounding_box_format="xywh",
        backbone=keras_cv.models.ResNet50(include_top=False, weights="imagenet", include_rescaling=True).as_backbone())
    return model

def unpackage_tfds_inputs(inputs):
    image = inputs["image"]
    boxes = bounding_box.convert_format(inputs["objects"]["bbox"], images=image, source="rel_yxyx", target="xywh")
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32)
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

def run(checkpoint_path, model):
    # get pascalvoc dataset
    train_dataset = tfds.load('voc/2007', 
        split='train+validation', 
        with_info=False, 
        shuffle_files=True,
        data_dir=data_directory)
    eval_dataset = tfds.load('voc/2007',
        split='test',
        with_info=False,
        data_dir=data_directory)
    # map to convert VOC dataset
    train_ds = train_dataset.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_dataset.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    optimizer = tf.optimizers.SGD(learning_rate=0.001)
    model.compile(
        classification_loss="focal",
        box_loss="smoothl1",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')

    #model.save_weights(checkpoint_path.format(epoch=0))

    start_time = time.time()
    model.fit(
        train_ds,
        epochs=epoch_count,
        validation_data=eval_ds,
        batch_size=batch_size,
        callbacks=[cp_callback]
    )
    end_time = time.time()

    # A final test to evaluate the model
    print('Test loss:', model.loss)
    print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
    print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))
    print(model.summary())


# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
