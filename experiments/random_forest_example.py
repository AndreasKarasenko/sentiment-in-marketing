from cuml.datasets.classification import make_classification as cuMK
import cuml
import time
from cupy import asnumpy
from joblib import dump, load
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification as skMK
from sklearn.ensemble import RandomForestClassifier as skRF

# synthetic dataset dimensions
def make_dataset( n_samples, n_features, n_classes ):
    X, y = skMK( n_samples = n_samples,
                 n_features = n_features,
                 n_classes = n_classes,
                 random_state = 0 )
    return X, y
n_samples = 1000
n_features = 10
n_classes = 2

# random forest depth and size
n_estimators = 25
max_depth = 10

# generate synthetic data [ binary classification task ]
X, y = cuMK ( n_classes = n_classes,
                             n_features = n_features,
                             n_samples = n_samples,
                             random_state = 0 )

print(X.shape, y.shape)
print(type(X))

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 0 )

model = cuRF( max_depth = max_depth,
              n_estimators = n_estimators,
              random_state  = 0 )

cutrain = time.time()
trained_RF = model.fit ( X_train, y_train )
cutrain = time.time() - cutrain

cupredict = time.time()
predictions = model.predict ( X_test )
cupredict = time.time() - cupredict

model = skRF( max_depth = max_depth,
              n_estimators = n_estimators,
              random_state  = 0 )

sktrain = time.time()
trained_RF = model.fit ( asnumpy( X_train ), asnumpy( y_train ) )
sktrain = time.time() - sktrain

skpredict = time.time()
predictions = model.predict ( asnumpy( X_test ) )
skpredict = time.time() - skpredict

print( " cuml train time : ", cutrain, " cuml predict time : ", cupredict )
print( " sklearn train time : ", sktrain, " sklearn predict time : ", skpredict )
print( " Difference in train time : ", cutrain / sktrain ) # below 1 is faster, above 1 is slower
print( " Difference in predict time : ", cupredict / skpredict ) # below 1 is faster, above 1 is slower
### Note that for very few samples it might be faster to run it on the CPU
### e.g. changing samples to 50,000 gives cuml a big advantage

cu_score = cuml.metrics.accuracy_score( y_test, predictions )
sk_score = accuracy_score( asnumpy( y_test ), asnumpy( predictions ) )

print( " cuml accuracy: ", cu_score )
print( " sklearn accuracy : ", sk_score )