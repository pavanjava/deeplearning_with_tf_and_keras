import tensorflow as tf
from tensorflow import keras


class SimplePerceptron:
    def __init__(self):
        self.NB_CLASSES = 10
        self.N_HIDDEN = 30
        self.RESHAPED = 28 * 28
        self.EPOCHS = 50
        self.BATCH_SIZE = 128
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.verbose = 1
        self.DROPOUT = 0.3

    def preprocess(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = self.x_train.reshape(len(self.x_train), self.RESHAPED)
        self.x_test = self.x_test.reshape(len(self.x_test), self.RESHAPED)

        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")

        self.x_train /= 255.0
        self.x_test /= 255.0

        self.y_train = keras.utils.to_categorical(self.y_train, self.NB_CLASSES)
        self.y_test = keras.utils.to_categorical(self.y_test, self.NB_CLASSES)

    '''
    kernel_initializer can take any one value from ["random_uniform", "random_normal", "zeros"]. the default in this case 
    is taken as "zeros"
    
    activation can take any one value from ["sigmoid", "tanh", "softmax", "relu", "leakyRelu"]. the default in this case
    is taken as "softmax"
    
    optimizer can take any value from ["adam", "SGD", "RMSProp"] default in this case is taken as "adam"
    
    loss can take any value from ["mse", "binary_crossentropy", "categorical_crossentropy"]. the default being
    "categorical_crossentropy"
    '''
    def create_model(self, end_layer_activation="softmax", optimizer="adam", loss="categorical_crossentropy"):
        self.model = tf.keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.N_HIDDEN, input_shape=(self.RESHAPED,),
                                          name="dense_layer_1", activation='relu'))
        self.model.add(keras.layers.Dropout(self.DROPOUT))
        self.model.add(keras.layers.Dense(self.N_HIDDEN, input_shape=(self.RESHAPED,),
                                           name="dense_layer_2",activation='relu'))
        self.model.add(keras.layers.Dropout(self.DROPOUT))
        self.model.add(keras.layers.Dense(self.NB_CLASSES, input_shape=(self.RESHAPED,),
                                          name="dense_layer_3", activation=end_layer_activation))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        print(self.model.summary())

    def fit(self):
        self.model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_split=0.2, verbose=self.verbose)

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"test accuracy: {test_acc}")


if __name__ == "__main__":
    obj = SimplePerceptron()
    obj.preprocess()
    obj.create_model()
    obj.fit()
    obj.evaluate()
