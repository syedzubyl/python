import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
data = np.array([[i for i in range(10)] for _ in range (100)])
X,y = data[:, :-1],data[:, -1]
X = X.reshape((X.shape[0], X.shape[1], 1))
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(9,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)
test_input = np.array([7,8,9,10,11,12,13,14,15])
test_input = test_input.reshape((1,9,1))
predicted_value = model.predict(test_input, verbose=0)
print(f'Predicted value: {predicted_value[0][0]}')
