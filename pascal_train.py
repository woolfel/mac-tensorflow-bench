# Disclaimer!!   this script doesn't work on Windows or Mac yet. Even though I borrowed from Keras_cv guides,
# this spits out a bunch of errors. The only difference is this vesion doesn't use linux specific libraries
# and skips the visualization steps. The original script is https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py
# Since google doesn't care about writing good documentation or keeping it up-to-date, developers have to suffer
# the pain of a million paper cuts.
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import optimizers
import keras_cv
from keras_cv import bounding_box
from keras_cv.callbacks import PyCOCOCallback
import time
import os
import sys

print('Tensorflow version - ',tf.__version__)
print('Keras_cv version - ', keras_cv.__version__)

checkpoint_path = "training/weights.{epoch:02d}-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
data_directory = 'H:/tensorflow_datasets/voc'
batch_size=64
epoch_count=50
num_classes=20

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
    model = keras_cv.models.RetinaNet(classes=num_classes,bounding_box_format="xywh",
        backbone=keras_cv.models.ResNet50(include_top=False, weights="imagenet", include_rescaling=True).as_backbone())
    model.backbone.trainable = False
    return model

def unpackage_tfds_inputs(inputs):
    image = inputs["image"]
    boxes = bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target="xywh",
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=float),
        "boxes": tf.cast(boxes, dtype=float),
    }
    return {"images": tf.cast(image, float), "bounding_boxes": bounding_boxes}


def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

def run(checkpoint_path, model):
    # get pascalvoc dataset
    train_dataset = tfds.load('voc/2007', split='train+validation', 
        with_info=False, shuffle_files=True, data_dir=data_directory)
    eval_dataset = tfds.load('voc/2007', split='test',
        with_info=False, data_dir=data_directory)
    # map to convert VOC dataset
    train_ds = train_dataset.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_dataset.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    class_ids = [
    "Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car",
    "Cat","Chair","Cow","Dining Table","Dog","Horse","Motorbike",
    "Person","Potted Plant","Sheep","Sofa","Train","Tvmonitor","Total"]

    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

    base_lr = 0.01
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[12000 * 16, 16000 * 16],
        values=[base_lr, 0.1 * base_lr, 0.01 * base_lr],)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        classification_loss="focal",
        box_loss="smoothl1",
        optimizer=optimizer)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')
    callbacks = [PyCOCOCallback(eval_ds, bounding_box_format="xywh"),
        keras.callbacks.TensorBoard(log_dir="logs"),
        cp_callback]

    start_time = time.time()
    model.fit(
        train_ds,
        epochs=epoch_count,
        validation_data=eval_ds,
        batch_size=batch_size,
        callbacks=callbacks
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
