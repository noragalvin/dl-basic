# Thêm thư viện
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json


# Lớp neural network
class MNIST_CNN:
    def __init__(self, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.Y_train = None
      self.Y_val = None
      self.Y_test = None

      self.model = Sequential()

      self.load_data()
      self.reshape_data()
      self.encoding_label()
      self.init_model()
      self.fit()
      self.save_model('model.json')

    def load_data(self):
      self.X_val, self.y_val = self.X_train[50000:60000,:], self.y_train[50000:60000]
      self.X_train, self.y_train = self.X_train[:50000,:], self.y_train[:50000]
      print(self.X_train.shape)

    def reshape_data(self):
      self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
      self.X_val = self.X_val.reshape(self.X_val.shape[0], 28, 28, 1)
      self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
    
    def encoding_label(self):
      self.Y_train = np_utils.to_categorical(self.y_train, 10)
      self.Y_val = np_utils.to_categorical(self.y_val, 10)
      self.Y_test = np_utils.to_categorical(self.y_test, 10)
      print('Du lieu y ban dau ', self.y_train[0])
      print('Du lieu y sau one-hot encoding ', self.Y_train[0])

    def init_model(self):
      # Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3
      # dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên
      self.model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28,28,1)))

      # Thêm Convolutional layer
      self.model.add(Conv2D(32, (3, 3), activation='sigmoid'))

      # Thêm Max pooling layer
      self.model.add(MaxPooling2D(pool_size=(2,2)))

      # Flatten layer chuyển từ tensor sang vector
      self.model.add(Flatten())

      # Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
      self.model.add(Dense(128, activation='sigmoid'))

      # Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.
      self.model.add(Dense(10, activation='softmax'))

      # 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức 
      # đùng để tối ưu hàm loss function.
      self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
    
	# Tóm tắt mô hình neural network
    def __repr__(self):
      return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
      
    def fit(self, batch_size=32, epochs=10, verbose=1):
      self.H = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), batch_size=batch_size, epochs=epochs, verbose=verbose)
    
    def load_model(self, file_name):
      # load json and create model
      json_file = open(file_name, 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      self.model = model_from_json(loaded_model_json)
    
    def predict(self, X):
      y_predict = self.model.predict(X.reshape(1,28,28,1))
      print('Gia tru du doan: ', np.argmax(y_predict))

    def save_model(self, file_name):
      # serialize model to JSON
      model_json = self.model.to_json()
      with open(file_name, "w") as json_file:
          json_file.write(model_json)
      print("Saved model to disk")


(X_train, y_train), (X_test, y_test) = mnist.load_data()

p = MNIST_CNN(X_train, y_train, X_test, y_test)
p.predict(X_test[0])