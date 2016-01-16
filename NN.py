from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization

class NN:
    #I made a small wrapper for the Keras model to make it more scikit-learn like
    #I think they have something like this built in already, oh well
    #See http://keras.io/ for parameter options
    def __init__(self, inputShape, layers, dropout = [], activation = 'relu', init = 'uniform', loss = 'rmse', optimizer = 'adadelta', nb_epochs = 50, batch_size = 32, verbose = 1):

        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                print ("Input shape: " + str(inputShape))
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], input_dim = inputShape, init = init))
            else:
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], init = init))
            print ("Adding " + activation + " layer")
            model.add(Activation(activation))
            model.add(BatchNormalization())
            if len(dropout) > i:
                print ("Adding " + str(dropout[i]) + " dropout")
                model.add(Dropout(dropout[i]))
        model.add(Dense(1, init = init)) #End in a single output node for regression style output
        model.compile(loss=loss, optimizer=optimizer)
        
        self.model = model
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y): 
        self.model.fit(X, y, nb_epoch=self.nb_epochs, batch_size=self.batch_size, verbose = self.verbose)
        
    def predict(self, X, batch_size = 128, verbose = 1):
        return self.model.predict(X, batch_size = batch_size, verbose = verbose)
