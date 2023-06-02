import keras_cv
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import time
import sys

print(tf.__version__)
print(keras_cv.__version__)

def main():
    args = sys.argv[0:]
    savepath = args[1] + "/weights.{epoch:02d}-{accuracy:.3f}-{loss:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
    epoch = int(args[2])
    batch_size = int(args[3])
    logpath = args[1] + "/rockpaperscissor_training.csv"
    train(savepath, epoch, batch_size, logpath)

def createModel():
    model = tf.keras.applications.ResNet50V2(
    include_top=False, weights=None,
    input_tensor=None, input_shape=(300,300,3),
    pooling="max", classifier_activation="softmax", classes=3)
    return model

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def createDataset(batchsize):
    (train, test), info = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    train = train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.cache()
    train = train.shuffle(info.splits['train'].num_examples)
    train = train.batch(batchsize)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    test = test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.batch(batchsize)
    test = test.cache()
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return (train, test), info

def train(savepath, epoch, batchsize, logpath):
    #create the model with the given python script
    model = createModel()
    #compile the model 
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train, test), info = createDataset(batchsize)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=savepath,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')
    csv_logger = tf.keras.callbacks.CSVLogger(logpath,append=True)

    start_time = time.time()

    model.fit(
        train,
        epochs=epoch,
        validation_data=test,
        batch_size=batchsize,
        callbacks=[cp_callback, csv_logger]
    )
    end_time = time.time()
    print('Test loss:', model.loss)
    print(model.summary())
    print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
    print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))

if __name__ == "__main__":
    main()