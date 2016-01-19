import numpy as np
import quadratic_weighted_kappa

class CutPointOptimizer:
    
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual

    def qwk(self, cutPoints):
        transformedPredictions = np.searchsorted(cutPoints, self.predicted) + 1            
        return -1 * quadratic_weighted_kappa.quadratic_weighted_kappa(transformedPredictions, self.actual)

