import pandas as pd 

def prediction_result(Xtest, ytest, ypred, target:str):
    submission = pd.DataFrame(Xtest)
    submission[target] = ytest
    submission[target + '-(Pred)'] = ypred
    return submission