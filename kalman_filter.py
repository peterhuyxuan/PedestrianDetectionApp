import numpy as np


class Kalman_Filter(object):
    #     u: previous state vector
    #     P: The state covariance of previous step ( k −1).
    #     F: The transition n * n matrix. or A 
    #     Q: process noise matrix
    #     A: matrix in observation equations
    #     b: centers for prediction
    #     R: observation noise matrix

    def __init__(self):
        
        self.delta_time = 0.004

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])

        self.P = np.diag((3.0, 3.0))
        self.F = np.array([[1.0, self.delta_time], [0.0, 1.0]])

        self.Q = np.eye(self.u.shape[0])  
        self.R = np.eye(self.b.shape[0])  
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        # Predict mean u and covariance P of the previous state.
        
        # state predicted
        self.u = np.round(np.dot(self.F, self.u))
        # covariance predicted
        # P= F*P*F' + Q 
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        # Predicted last result
        self.lastResult = self.u  
        return self.u

    def update(self, b, flag):
        # C = A*P*A'+R 
        # K = P * A'* inv(A*P*A'+R)
        # update state vector u and covariance of uncertainty P.
        # Flag : True - update using prediction else update using detection
        # return : predicted state
        
        # update using prediction
        if not flag:  
            self.b = self.lastResult
        # update using detection
        else:  
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u
