""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """
"Tools for evaluating system identification"

import copy, itertools

import numpy as np
from scipy import linalg
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error


def computeEigIdError(trueEigs, idEigVals, measure='NFN'):
    permut = itertools.permutations(trueEigs)

    def computeErr(trueValue, prediction, measure):
        if measure == 'NFN':
            perf = np.sqrt(np.sum(np.abs(prediction-trueValue)**2, axis=1)) / np.sqrt(np.sum(np.abs(trueValue)**2, axis=1))
        else:
            raise('Not supported')
        return perf
    
    allEigError = np.array([])
    for i in range(len(idEigVals)):
        eigVals = idEigVals[i]
        if len(eigVals) > 0:
            errorVals = np.array([])
            for p in permut:
                errorVals = np.append( errorVals, computeErr( np.atleast_2d(p), np.atleast_2d(eigVals), measure) )

            pmi = np.argmin( errorVals )
            allEigError = np.append( allEigError, errorVals[pmi] )
    return allEigError

def computeLSSMIdError(sTrue, sId_in):
    errs = {}
    if hasattr(sId_in, 'state_dim') and sTrue.state_dim == sId_in.state_dim:
        sId = copy.deepcopy(sId_in) # Do not modify the original object
        E = sId.makeSimilarTo(sTrue)
        def matrixErrNorm(trueX, idX):
            errX = idX-trueX
            return np.linalg.norm(errX, ord='fro') / np.linalg.norm(trueX, ord='fro')

        for field in dir(sTrue): 
            valTrue = sTrue.__getattribute__(field)
            if not field.startswith('__') and isinstance(valTrue, np.ndarray):
                if field in dir(sId):
                    valId = sId.__getattribute__(field)
                    if isinstance(valId, np.ndarray) and valTrue.shape == valId.shape:
                        errFieldName = '{}ErrNormed'.format(field)
                        try:
                            errs[errFieldName] = matrixErrNorm(np.atleast_2d(valTrue), np.atleast_2d(valId))
                        except Exception as e:
                            print(e)
                            pass
            
    # Eigenvalue error
    nz = len(sTrue.zDims)
    if hasattr(sId_in, 'zDims') and len(sId_in.zDims) > 0:
        subATrue = sTrue.A[np.ix_(np.array(sTrue.zDims)-1, np.array(sTrue.zDims)-1)]
        subAId   = sId_in.A[np.ix_(np.array(sId_in.zDims)-1, np.array(sId_in.zDims)-1)]
        trueZEigs = linalg.eig(subATrue)[0]
        idZEigs = linalg.eig(subAId)[0]
        errs['zEigErrNFN'] = computeEigIdError(trueZEigs, [idZEigs], 'NFN')[0]
    
    return errs

def evaluateDecoding(sId, YTest, ZTest):
    zPredTest, yPredTest, xPredTest = sId.predict(YTest)
    
    errs = {}
    
    measures = ['CC', 'NRMSE', 'EV', 'R2']
    for m in measures:
        errs[m] = evalPrediction(ZTest, zPredTest, m)
        errs['mean'+m] = np.mean(errs[m])

        errs['y'+m] = evalPrediction(YTest, yPredTest, m)
        errs['meany'+m] = np.mean(errs['y'+m])

    return errs

def evalPrediction(trueValue, prediction, measure):

    if measure == 'CC':
        n = trueValue.shape[1]
        R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[n:, :n])
    elif measure == 'MSE':
        perf = mean_absolute_error(trueValue, prediction, multioutput='raw_values')
    elif measure == 'RMSE':
        MSE = mean_squared_error(trueValue, prediction, multioutput='raw_values') # squared=False doesn't work for multioutput='raw_values'!
        perf = np.sqrt(MSE)
    elif measure == 'NRMSE':
        RMSE = evalPrediction(trueValue, prediction, 'RMSE')
        perf = RMSE/np.std(trueValue, axis=0)
    elif measure == 'MAE':
        perf = mean_absolute_error(trueValue, prediction, multioutput='raw_values')
    elif measure == 'NMAE':
        MAE = evalPrediction(trueValue, prediction, 'MAE')
        perf = MAE/mean_absolute_error(trueValue, np.zeros_like(prediction), multioutput='raw_values')
    elif measure == 'EV':
        perf = explained_variance_score(trueValue, prediction, multioutput='raw_values')
    elif measure == 'R2':
        perf = r2_score(trueValue, prediction, multioutput='uniform_average')
    return perf
