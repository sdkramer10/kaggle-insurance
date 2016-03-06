from scipy import optimize
from CutPoints import CutPointOptimizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from sklearn.cross_validation import train_test_split
from keras.callbacks import Callback
import numpy as np
import quadratic_weighted_kappa
from keras.layers.advanced_activations import PReLU
import theano
import theano.tensor as T

def rounded_mean_absolute_error(y_true, y_pred):
    rounded = T.round(y_pred)
    return T.mean(T.abs_(rounded - y_true), axis=-1)

class clsvalidation_kappa(Callback):  #inherits from Callback
    
    def __init__(self, filepath, train_data=(), validation_data=(), patience=5):
        super(Callback, self).__init__()

        self.patience = patience
        self.X_val, self.y_val = validation_data 
        self.X_train, self.Y_train = train_data
        self.best = 0.0
        self.wait = 0  #counter for patience
        self.filepath=filepath
        self.best_rounds =1
        self.counter=0
        self.cutPoints = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    
    def on_epoch_end(self, epoch, logs={}):
        
        self.counter +=1
        train_predictions = self.model.predict(self.X_train, verbose=0)
        cpo = CutPointOptimizer(train_predictions, self.Y_train)
        self.cutPoints = optimize.fmin(cpo.qwk, self.cutPoints)
        
        p = self.model.predict(self.X_val, verbose=0) #score the validation data 
        
        p = np.searchsorted(self.cutPoints, p) + 1   
        current = quadratic_weighted_kappa.quadratic_weighted_kappa(self.y_val.values.ravel(), p)       

        print('Epoch %d Kappa: %f | Best Kappa: %f \n' % (epoch,current,self.best))
    
    
        #if improvement over best....
        if current > self.best:
            self.best = current
            self.best_rounds=self.counter
            self.wait = 0
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.wait >= self.patience: #no more patience, retrieve best model
                self.model.stop_training = True
                print('Best number of rounds: %d \nKappa: %f \n' % (self.best_rounds, self.best))
                
                self.model.load_weights(self.filepath)
                           
            self.wait += 1 #incremental the number of times without improvement

class NN:
    #I made a small wrapper for the Keras model to make it more scikit-learn like
    #I think they have something like this built in already, oh well
    #See http://keras.io/ for parameter options
    def __init__(self, inputShape, cutPointArray, layers, dropout = [], activation = 'relu', patience=5, init = 'uniform', loss = 'rmse', optimizer = 'adadelta', nb_epochs = 50, batch_size = 32, verbose = 1):

        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                print ("Input shape: " + str(inputShape))
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], input_dim = inputShape, init = init))
            else:
                print ("Adding Layer " + str(i) + ": " + str(layers[i]))
                model.add(Dense(layers[i], init = init))
            #model.add(Activation(activation))
            model.add(PReLU())
            model.add(BatchNormalization())
            if len(dropout) > i:
                print ("Adding " + str(dropout[i]) + " dropout")
                model.add(Dropout(dropout[i]))
        model.add(Dense(1, init = init)) #End in a single output node for regression style output
        model.compile(loss=loss, optimizer=optimizer)
        self.optimizer = optimizer
        self.loss = loss    
        self.model = model
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.patience = patience
        self.cutPointArray = cutPointArray

    def fit(self, X, y): 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
        cutPoints = np.median(self.cutPointArray, axis=0)
        val_call = clsvalidation_kappa(train_data=(X_train, y_train), validation_data=(X_val, y_val), patience=self.patience, filepath='../input/best.h5') 
        self.model.fit(X_train, y_train, nb_epoch=self.nb_epochs, batch_size=self.batch_size, verbose = self.verbose, callbacks=[val_call])
        
    def predict(self, X, batch_size = 128, verbose = 1):
        return self.model.predict(X, batch_size = batch_size, verbose = verbose)[:,0]

