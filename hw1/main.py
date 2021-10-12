import loader
import pandas as pd
import numpy as np
import random
import numpy.linalg as la
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dataset1 = loader.loadNormalizedDataset('dataset1.csv')
test_dataset1 = loader.loadNormalizedDataset('test_dataset1.csv')

dataset2 = loader.loadNormalizedDataset('dataset2.csv')
test_dataset2 = loader.loadNormalizedDataset('test_dataset2.csv')

dataset3 = loader.loadNormalizedDataset('dataset3.csv')
test_dataset3 = loader.loadNormalizedDataset('test_dataset3.csv')

dataset4 = loader.loadNormalizedDataset('dataset4.csv')
test_dataset4 = loader.loadNormalizedDataset('test_dataset4.csv')

dataset5 = loader.loadNormalizedDataset('dataset5.csv')
test_dataset5 = loader.loadNormalizedDataset('test_dataset5.csv')

# Separate dataset to X* and Y
def separateXY(df: pd.DataFrame):
    col = len(df.columns)
    X = df.iloc[:, :col - 1]
    Y = df.iloc[:, col - 1:]
    return X, Y

def insertBias(df: pd.DataFrame):
    df.insert(0, 'bias', 1, True)

X1, Y1 = separateXY(dataset1)
insertBias(X1)
X1 = X1.to_numpy()
Y1 = Y1.to_numpy()

TX1, TY1 = separateXY(test_dataset1)
insertBias(TX1)
TX1 = TX1.to_numpy()
TY1 = TY1.to_numpy()

X2, Y2 = separateXY(dataset2)
insertBias(X2)
X2 = X2.to_numpy()
Y2 = Y2.to_numpy()

TX2, TY2 = separateXY(test_dataset2)
insertBias(TX2)
TX2 = TX2.to_numpy()
TY2 = TY2.to_numpy()

X3, Y3 = separateXY(dataset3)
insertBias(X3)
X3 = X3.to_numpy()
Y3 = Y3.to_numpy()

TX3, TY3 = separateXY(test_dataset3)
insertBias(TX3)
TX3 = TX3.to_numpy()
TY3 = TY3.to_numpy()

X4, Y4 = separateXY(dataset4)
insertBias(X4)
X4 = X4.to_numpy()
Y4 = Y4.to_numpy()

TX4, TY4 = separateXY(test_dataset4)
insertBias(TX4)
TX4 = TX4.to_numpy()
TY4 = TY4.to_numpy()

X5, Y5 = separateXY(dataset5)
insertBias(X5)
X5 = X5.to_numpy()
Y5 = Y5.to_numpy()

TX5, TY5 = separateXY(test_dataset5)
insertBias(TX5)
TX5 = TX5.to_numpy()
TY5 = TY5.to_numpy()

# Gradient of MSE
def GradMSE(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    pred = X.dot(w)
    _, col = X.shape
    return (2 / col) * X.T.dot(pred - Y)

# Step function
init_step = 0.0001
def step(i):
    return init_step / (i)

# Gradient Decscsdcds
col1 = len(dataset1.columns)
col2 = len(dataset2.columns)
col3 = len(dataset3.columns)
col4 = len(dataset4.columns)
col5 = len(dataset5.columns)

w1 = np.array([[random.randint(-10, 10) * 0.001] for i in range(col1)])
w2 = np.array([[random.randint(-10, 10) * 0.001] for i in range(col2)])
w3 = np.array([[random.randint(-10, 10) * 0.001] for i in range(col3)])
w4 = np.array([[random.randint(-10, 10) * 0.001] for i in range(col4)])
w5 = np.array([[random.randint(-10, 10) * 0.001] for i in range(col4)])

def teach(X: np.ndarray, Y: np.ndarray, w: np.ndarray, stepf, eps: float) -> np.ndarray:
    i = 1
    eps = 0.0001
    w_new = w - (step(i) * GradMSE(X, Y, w))
    norm = la.norm(w - w_new)
    flag = True
    while flag:
        print(i, norm)
        if norm < eps:
            flag = False
        w_new = w - (step(i) * GradMSE(X, Y, w))
        norm = la.norm(w - w_new)
        w = w_new
        i = i + 1
    return w_new

ans1 = teach(X1, Y1, w1, step, 0.001)
# pd.DataFrame(ans1).to_csv('ans1.csv', header=False, index=False)
ans2 = teach(X2, Y2, w2, step, 0.001)
# pd.DataFrame(ans2).to_csv('ans2.csv', header=False, index=False)
ans3 = teach(X3, Y3, w3, step, 0.001)
# pd.DataFrame(ans3).to_csv('ans3.csv', header=False, index=False)
ans4 = teach(X4, Y4, w4, step, 0.001)
# pd.DataFrame(ans4).to_csv('ans4.csv', header=False, index=False)
ans5 = teach(X5, Y5, w5, step, 0.001)
# pd.DataFrame(ans5).to_csv('ans5.csv', header=False, index=False)

test1_mse = mean_squared_error(TY1, TX1.dot(ans1), squared=False)
test1_r2 = r2_score(TY1, TX1.dot(ans1))
print("FOLD 1 RMSE ON TEST:", test1_mse)
print("FOLD 1 R^2 ON TEST:", test1_r2)

train1_mse = mean_squared_error(Y1, X1.dot(ans1), squared=False)
train1_r2 = r2_score(Y1, X1.dot(ans1))
print("FOLD 1 RMSE ON TRAIN:", train1_mse)
print("FOLD 1 R^2 ON TRAIN:", train1_r2)

test2_mse = mean_squared_error(TY2, TX2.dot(ans2), squared=False)
test2_r2 = r2_score(TY2, TX2.dot(ans2))
print("FOLD 2 RMSE ON TEST:", test2_mse)
print("FOLD 2 R^2 ON TEST:", test2_r2)

train2_mse = mean_squared_error(Y2, X2.dot(ans2), squared=False)
train2_r2 = r2_score(Y2, X2.dot(ans2))
print("FOLD 2 RMSE ON TRAIN:", train2_mse)
print("FOLD 2 R^2 ON TRAIN:", train2_r2)

test3_mse = mean_squared_error(TY3, TX3.dot(ans3), squared=False)
test3_r2 = r2_score(TY3, TX3.dot(ans3))
print("FOLD 3 RMSE ON TEST:", test3_mse)
print("FOLD 3 R^2 ON TEST:", test3_r2)

train3_mse = mean_squared_error(Y3, X3.dot(ans3), squared=False)
train3_r2 = r2_score(Y3, X3.dot(ans3))
print("FOLD 3 RMSE ON TRAIN:", train3_mse)
print("FOLD 3 R^2 ON TRAIN:", train3_r2)

test4_mse = mean_squared_error(TY4, TX4.dot(ans4), squared=False)
test4_r2 = r2_score(TY4, TX4.dot(ans4))
print("FOLD 4 RMSE ON TEST:", test4_mse)
print("FOLD 4 R^2 ON TEST:", test4_r2)

train4_mse = mean_squared_error(Y4, X4.dot(ans4), squared=False)
train4_r2 = r2_score(Y4, X4.dot(ans4))
print("FOLD 4 RMSE ON TRAIN:", train4_mse)
print("FOLD 4 R^2 ON TRAIN:", train4_r2)

test5_mse = mean_squared_error(TY5, TX5.dot(ans5), squared=False)
test5_r2 = r2_score(TY5, TX5.dot(ans5))
print("FOLD 5 RMSE ON TEST:", test5_mse)
print("FOLD 5 R^2 ON TEST:", test5_r2)

train5_mse = mean_squared_error(Y5, X5.dot(ans5), squared=False)
train5_r2 = r2_score(Y5, X5.dot(ans5))
print("FOLD 5 RMSE ON TRAIN:", train5_mse)
print("FOLD 5 R^2 ON TRAIN:", train5_r2)

test_rmse_array = np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse])
print("RMSE ON TEST MEAN:", np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse]).mean())
print("RMSE ON TEST STD:", np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse]).std())

test_r2_array = np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2])
print("R2 ON TEST MEAN:", np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2]).mean())
print("R2 ON TEST STD:", np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2]).std())

train_rmse_array = np.array([train1_mse, train2_mse, train3_mse, train4_mse,train5_mse])
print("RMSE ON TRAIN MEAN:", np.array([train1_mse, train2_mse, train3_mse, train4_mse,train5_mse]).mean())
print("RMSE ON TRAIN STD:", np.array([train1_mse, train2_mse, train3_mse, train4_mse, train5_mse]).std())

train_r2_array = np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2])
print("R2 ON TRAIN MEAN:", np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2]).mean())
print("R2 ON TRAIN STD:", np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2]).std())