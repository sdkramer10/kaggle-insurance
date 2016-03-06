import numpy as np
import xgboost as xgb
import quadratic_weighted_kappa
import numpy as np
from scipy import optimize
from CutPoints import CutPointOptimizer
from sklearn.cross_validation import train_test_split

class XGBoostModel:
    
    def __init__(self, num_rounds, max_depth, eta, colsample_bytree, min_child_weight, silent=True):
        self.param = {'max_depth':max_depth, 'eta': eta, 'silent':1, 'min_child_weight':min_child_weight, 'subsample' : 0.7,
              'colsample_bytree': colsample_bytree, "silent" : silent}
        self.num_round=num_rounds
        self.cpa = np.ndarray((7, 7))
        self.cpa[0,:] = [2.68237582,  3.38457017,  4.26592497,  4.89213188,  5.59007684,  6.30401051, 6.86168173]
        self.cpa[1,:] = [ 1.83758628,  3.28546717,  4.30051656,  5.00993717,  5.54911030,  6.2537504, 6.93040581]
        self.cpa[2,:] = [ 2.19477416,  3.28905817,  4.27904599,  4.90782139,  5.67257616,  6.16319723, 6.86728276]
        self.cpa[3,:] = [ 1.68366001,  3.45431677,  4.20001421,  4.93482508,  5.54030709,  6.25156745, 6.88775039]
        self.cpa[4,:] = [ 1.61531459,  3.08294951,  4.39483963,  4.73639875,  5.48588959,  6.20398412, 6.73928103]
        self.cpa[5,:] = [ 1.96627584,  3.54029862,  4.13203389,  4.80030722,  5.5207599,   6.25109169, 6.85618128]
        self.cpa[6,:] = [ 1.66802702,  3.46534344,  3.89126679,  4.81654785,  5.35745692,  6.28049139, 6.89256955]
        self.cutPoints = [ 1.91843059,  3.29717288,  4.208207060,  4.86836428,  5.46248675,  6.2095608, 6.86633969]
        
    def fit(self, xTrain, yTrain, fold):
        #self.cutPoints = self.cpa[fold,:]
        #X_train, X_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.12, random_state=107)
        dtrain = xgb.DMatrix(xTrain,label=yTrain)
        #xgtrain = xgb.DMatrix(X_train, label=y_train)
        #xgval=xgb.DMatrix(X_test,label=y_test)        
        watchlist  = [(dtrain,'train')]
        self.bst = xgb.train(self.param, dtrain, self.num_round, watchlist, obj=kapparegobj, feval=self.qwkerror)
        
    def predict(self, testData):
        dTest = xgb.DMatrix(testData)
        predictions =  self.bst.predict(dTest, ntree_limit=self.bst.best_ntree_limit)
        return -0.514476838741 + 1.05951888 * predictions

    def qwkerror(self, preds, dtrain):
        labels = dtrain.get_label()
        preds = np.searchsorted(self.cutPoints, preds) + 1 
        kappa = quadratic_weighted_kappa.quadratic_weighted_kappa(labels, preds)
        return 'kappa', -1 * kappa

# custom cost function - similar to rmse, but take into account the fact the closer a prediction is to the truth,
# the more likely it is that the cut points will cause it to get rounded to the correct truth value
def kapparegobj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    grad = 2*x*np.exp(-(x**2))*(np.exp(x**2)+x**2+1)
    hess = 2*np.exp(-(x**2))*(np.exp(x**2)-2*(x**4)+5*(x**2)-1)
    return grad, hess

def kappaerror(preds, dtrain):
    labels = dtrain.get_label()
    x = (labels-preds)
    error = (x**2)*(1-np.exp(-(x**2)))
    return 'error', np.mean(error)
