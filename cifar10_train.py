import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

print(tf.__version__)

# The filename format has the epoch number + accuracy + loss in HDF5 format
# the reason for using HDF5 format is cross platform compatibility and make it easier to load in other languages
checkpoint_path = "training/weights.{epoch:02d}-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 64

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

    run(checkpoint_path, batch_size)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def run(savepath, batch):
    # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # you can change the batch size to see how it performs. Larger batch size will stress GPU more
    epoch_count = 25

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # this version turns off bias for the first 2 hidden layers. Turning off bias
    # reduces the chance of overfitting, but it takes more epochs to reach the
    # same level of accuracy than with bias on
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', name='L1_conv2d', use_bias=False),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', name='L2_conv2d', use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L3_MaxP'),
    tf.keras.layers.Conv2D(256, kernel_size=(1, 1), activation='relu', name='L4_conv2d'),
    tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', name='L5_conv2d'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L6_MaxP'),
    tf.keras.layers.Conv2D(256, kernel_size=(1, 1), activation='relu', name='L7_conv2d'),
    tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', name='L8_conv2d'),
    tf.keras.layers.Dropout(0.290, name='L9_Drop'),
    tf.keras.layers.Flatten(name='L10_flat'),
    tf.keras.layers.Dense(128, activation='relu', name='L11_Dense'),
    tf.keras.layers.Dropout(0.5683, name='L12_Drop'),
    tf.keras.layers.Dense(10, activation='softmax', name='Dense_output')
    ], "cifar-test-3")

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=savepath,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')

    #model.save_weights(checkpoint_path.format(epoch=0))

    start_time = time.time()

    model.fit(
        ds_train,
        epochs=epoch_count,
        validation_data=ds_test,
        batch_size=batch,
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
