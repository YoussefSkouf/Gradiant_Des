import numpy as np
import copy


def predicted_value(x, w, b):
    return np.dot(x, w_init) + b

def cost_function(X, w, b, reel_value):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        cost = cost +(predicted_value(X[i], w_init, b_init) - reel_value[i])**2  
    return cost/(2*m)

def gradient (X, y, w_init, b_init):
       
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        error = predicted_value(X[i], w_init, b_init) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i,j]
        dj_db = dj_db + error
    return dj_dw/m,dj_db/m
    
def gradient_desc(X, y, w, b, alfa, num_itr):
    w = copy.deepcopy(w)  #avoid modifying global w within function
    b = b
    for i in range(num_itr):
        dw_dj, dw_db = gradient(X, y, w, b)
        
        w = w - alfa * dw_dj
        b = b - alfa * dw_db
    
    return w, b       
    
# Updated training data
X_train = np.array([[2500, 4, 2, 50], [1500, 3, 2, 45], [900, 2, 1, 30]])
y_train = np.array([500, 300, 200])

# Updated initial parameters
b_init = 1000.0
w_init = np.array([0.5, 15.0, -45.0, -20.0])
   
initial_w = np.zeros_like(w_init)
initial_b = 0.

# Updated gradient descent settings
iterations = 500
alpha = 1.0e-7

# run gradient descent 
w_final, b_final= gradient_desc(X_train, y_train, initial_w, initial_b,
                                                     alpha, iterations) 

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    
print(predicted_value(X_train[0], w_init, b_init))
print(cost_function(X_train, w_init, b_init, y_train))
    
tmp_dj_db, tmp_dj_dw = gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')



