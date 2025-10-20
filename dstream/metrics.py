from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def mean_error(y_test,y_pred):
    MSE= mean_squared_error(y_test,y_pred)
    R2 = r2_score(y_test,y_pred)
    print(f'Mean squared error:{MSE:.2f}\nCoefficient of determination(R^2):{R2:.2f}')