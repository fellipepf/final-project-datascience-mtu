import numpy as np
from keras.layers import Dense
from keras.models import Sequential


model = Sequential()
model.add(Dense(units=16,activation='relu',input_dim=2))   # (current layer* input) + ( bias* current layer) = (16 *2) + (1*16) = 48
model.add(Dense(units=1,activation='sigmoid'))   #(1*16) + (1*1) = 17
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])

print(model.get_weights())

x = np.array([[0.,0.],
              [0.,1.],
              [1.,0.],
              [1.,1.]])

y = np.array([0.,1.,1.,0.])

model.fit(x,y,epochs=1000, verbose=2)

print(model.summary())

print(model.get_weights())
print(model.predict(x))