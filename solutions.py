
## STUDENT ID: FILL IN YOUR ID
## STUDENT NAME: FILL IN YOUR NAME


## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
plt.scatter(x,y)
plt.show()

## (c)

# YOUR CODE HERE
w0=1
w1=1
c=2
alphas = [10e-1, 10e-2, 10e-3,10e-4,10e-5,10e-6,10e-7, 10e-8, 10e-9]
instance_diff_w0 = np.zeros(100)
instance_diff_w1 = np.zeros(100)
updated_loss = np.zeros([9,100])
losses = []
for i in range(9):
    w0=1
    w1=1
    for j in range(100):
        for p in range(100):
            instance_diff_w0[p] = (2*w0 - 2*y[p] + 2*w1*x[p])/(2*c**2*np.sqrt(((w0 - y[p] + w1*x[p])**2/c**2 + 1)))
            instance_diff_w1[p] = (x[p]*(w0 - y[p] + w1*x[p]))/(c**2*np.sqrt(((w0 - y[p] + w1*x[p])**2/c**2 + 1)))
        sum_diff_w0 = np.sum(instance_diff_w0)
        sum_diff_w1 = np.sum(instance_diff_w1)
        w0 = w0-alphas[i]*sum_diff_w0
        w1 = w1-alphas[i]*sum_diff_w1 
        loss = np.sum(np.sqrt((1/c**2)*(y-w0-w1*x)**2+1)-1)
        updated_loss[i,j] = loss
    losses.append(updated_loss[i,:])
    
## plotting help
fig, ax = plt.subplots(3,3, figsize=(10,10))
alphas = [10e-1, 10e-2, 10e-3,10e-4,10e-5,10e-6,10e-7, 10e-8, 10e-9]
for i, ax in enumerate(ax.flat):
    # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
    ax.plot(losses[i])         
    ax.set_title(f"step size: {alphas[i]}")	 # plot titles	
plt.tight_layout()      # plot formatting
plt.show()

## (e)

weights_0 = []
weights_1 = []
alphas= 0.01
for j in range(100):
    for p in range(100):
        instance_diff_w0[p] = (2*w0 - 2*y[i] + 2*w1*x[i])/(2*c**2*np.sqrt(((w0 - y[i] + w1*x[i])**2/c**2 + 1)))
        instance_diff_w1[p] = (x[i]*(w0 - y[i] + w1*x[i]))/(c**2*np.sqrt(((w0 - y[i] + w1*x[i])**2/c**2 + 1)))
    sum_diff_w0 = np.sum(instance_diff_w0)
    sum_diff_w1 = np.sum(instance_diff_w1)
    w0 = w0-alphas*sum_diff_w0
    w1 = w1-alphas*sum_diff_w1 
    weights_0.append(w0)
    weights_1.append(w1)
y_hat = w0+w1*x

plt.plot(x, weights_0, label = "w0")
plt.plot(x,weights_1, label = "w1")
plt.legend()
plt.show()
plt.plot(x, y_hat,'r',label='y_hat')
plt.scatter(x,y,label='Ground Truth')
plt.legend()
plt.savefig('pred_y.png')
plt.show()

## Question 3
import numpy as np
from matplotlib import pyplot as plt

X = np.asarray([[-0.8,1],[3.9,0.4],[1.4,1],[0.1,-3.3],[1.2,2.7],[-2.45,0.1],[-1.5,0.5],[1.2,-1.5]])
y = np.asarray([1,-1,1,-1,-1,-1,1,1])
inx = [True if item == 1 else False for item in y]
n_inx = [False if item == 1 else True for item in y]
X1 = X[inx]
X0 = X[n_inx]
plt.plot(X1[:,0],X1[:,1],'r.')
plt.plot(X0[:,0],X0[:,1],'b.')
inner_product = np.matmul(X,X.T)
M = (inner_product)**2 + 0
print(M[0,2], M[0,6], M[0,7], M[2,6], M[2,7], M[6,7])
print(M[1,3], M[1,4], M[1,5], M[3,4], M[3,5], M[4,5])
X2 = np.zeros((X.shape[0],3))
X2[:,0] = X[:,0] **2
X2[:,1] = X[:,1] **2
X2[:,2] = X[:,0] * X[:,1]

inx = [True if item == 1 else False for item in y]
n_inx = [False if item == 1 else True for item in y]
X_pos = X2[inx]
X_neg = X2[n_inx]

plt.plot(X_pos[:,0],X_pos[:,1],'r.')
plt.plot(X_neg[:,0],X_neg[:,1],'b.')

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

activation = sum(weight_i * x_i) + bias
prediction = 1.0 if activation >= 0.0 else 0.0
w = w + learning_rate * (expected - predicted) * x




# Question 5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y


# (a)
# YOUR CODE HERE
X, y = create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
inx = [True if item == 1 else False for item in y]
n_inx = [False if item == 1 else True for item in y]
X1 = X[inx]
X0 = X[n_inx]
plt.plot(X1[:,0],X1[:,1],'r.')
plt.plot(X0[:,0],X0[:,1],'b.')

model_svc = SVC()
model_svc.fit(X_train, y_train)
plotter(model_svc , X, X_test , y_test , 'Default SVC' , ax=None)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
plotter(model_lr , X, X_test , y_test , 'Default Logistic Regression' , ax=None)

model_ada = AdaBoostClassifier()
model_ada.fit(X_train, y_train)
plotter(model_ada , X, X_test , y_test , 'Default Random Forest' , ax=None)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
plotter(model_rf , X, X_test , y_test , 'Default Random Forest' , ax=None)

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
plotter(model_dt , X, X_test , y_test , 'Default Decision Tree' , ax=None)

model_mlp = MLPClassifier()
model_mlp.fit(X_train, y_train)
plotter(model_mlp , X, X_test , y_test , 'Default MLP' , ax=None)

def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                            np.arange(y_min, y_max, plot_step)) 
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)


# (b)
# YOUR CODE HERE
from time import time

### DT
DT_accs, DT_time = [], []
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = DecisionTreeClassifier()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    DT_accs.append(mean_acc)
    DT_time.append(sum_time)
    
## RF
RF_accs, RF_time = [], []
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = RandomForestClassifier()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    RF_accs.append(mean_acc)
    RF_time.append(sum_time)

## Adaboost
Ada_accs, Ada_time = [],[]
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = AdaBoostClassifier()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    Ada_accs.append(mean_acc)
    Ada_time.append(sum_time)

## Logistic Regression
LR_accs, LR_time = [],[]
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = LogisticRegression()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    LR_accs.append(mean_acc)
    LR_time.append(sum_time)
    
## MLP
NN_accs, NN_time = [],[]
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = MLPClassifier()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    NN_accs.append(mean_acc)
    NN_time.append(sum_time)

## SVM
svm_accs, svm_time = [],[]
for size in range(50,1001,50):
    tmp_accs, tmp_time = [], []
    for i in range(10):
        st = time()

        inx = np.random.randint(X_train.shape[0], size=size)
        Xtr, ytr = X_train[inx, :], y_train[inx]
        
        model = SVC()
        model.fit(Xtr, ytr)
        
        pred = model.predict(X_test)
        acc = np.sum(pred == y_test) / len(pred)
        tmp_accs.append(acc)
        tmp_time.append(time() - st)
        
    mean_acc = np.mean(tmp_accs)
    sum_time = np.sum(tmp_time)
    svm_accs.append(mean_acc)
    svm_time.append(sum_time)
plt.plot(range(50,1001,50),DT_accs,'o-',color = 'blue')
plt.plot(range(50,1001,50),RF_accs,'o-',color = 'orange')
plt.plot(range(50,1001,50),Ada_accs,'o-',color = 'lime')
plt.plot(range(50,1001,50),LR_accs,'o-',color = 'red')
plt.plot(range(50,1001,50),NN_accs,'o-',color = 'darkred')
plt.plot(range(50,1001,50),svm_accs,'o-',color = 'chocolate')
plt.legend(['Decision Tree','Random Forest','Adaboost','Logistic Regression','Neural Network','SVM'],bbox_to_anchor=(1.03, 1))
_ = plt.title('Comparison of accuracies over models')

# (c)
# YOUR CODE HERE
plt.plot(range(50,1001,50),DT_time,'o-',color = 'blue')
plt.plot(range(50,1001,50),RF_time,'o-',color = 'orange')
plt.plot(range(50,1001,50),Ada_time,'o-',color = 'lime')
plt.plot(range(50,1001,50),LR_time,'o-',color = 'red')
plt.plot(range(50,1001,50),NN_time,'o-',color = 'darkred')
plt.plot(range(50,1001,50),svm_time,'o-',color = 'chocolate')
plt.legend(['Decision Tree','Random Forest','Adaboost','Logistic Regression','Neural Network','SVM'],bbox_to_anchor=(1.03, 1))
_ = plt.title('Comparison of runing time over models')