import tensorflow as tf
from tensorflow import keras


class SimplePerceptron:
    def __init__(self):
        self.NB_CLASSES = 10
        self.RESHAPED = 28 * 28
        self.EPOCHS = 10
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_flatten = None
        self.x_test_flatten = None
        self.model = None

    def preprocess(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        self.x_train /= 255
        self.x_test /= 255
        self.x_train_flatten = self.x_train.reshape(len(self.x_train), self.RESHAPED)
        self.x_test_flatten = self.x_test.reshape(len(self.x_test), self.RESHAPED)
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
    def create_model(self, kernel_initializer="zeros", activation="softmax", optimizer="adam", loss="categorical_crossentropy"):
        self.model = tf.keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.NB_CLASSES, input_shape=(self.RESHAPED,),
                                          kernel_initializer=kernel_initializer, name="dense_layer",
                                          activation=activation))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        print(self.model.summary())

    def fit(self):
        self.model.fit(self.x_train_flatten, self.y_train, epochs=self.EPOCHS)


if __name__ == "__main__":
    obj = SimplePerceptron()
    obj.preprocess()
    obj.create_model()
    obj.fit()
