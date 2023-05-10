#import random
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


class prediction_block():
    def __init__(self, prediction_type, ML_model_block):
        self.prediction_type = prediction_type
        self.clf = LogisticRegression(random_state=0).fit(ML_model_block.X, ML_model_block.Y)
    
    def update(self, ML_model_block):
        self.clf = LogisticRegression(random_state=0).fit(ML_model_block.X, ML_model_block.Y)
    
    def p(self, x):
        if (self.prediction_type == 'log_reg'):
            p = self.clf.predict([[x]])[0]
        elif (self.prediction_type == 'sigmoid'):
            p = expit(x * self.clf.coef_ + self.clf.intercept_).ravel()
        return p
#         if (x>0.5):
#             return 1
#         else:
#             return 0