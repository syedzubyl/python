
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
bottleneck = 32

input_image = tf.keras.layers.Input(shape=(784,))
encoded_input = tf.keras.layers.Dense(bottleneck, activation='relu')(input_image)
decoded_output = tf.keras.layers.Dense(784, activation='sigmoid')(encoded_input)
autoencoder = tf.keras.models.Model(input_image, decoded_output)
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))

autoencoder.fit(X_train,X_train,epochs = 30,batch_size = 256, shuffle = True, validation_data =
(X_test, X_test))
reconstructed_img = autoencoder.predict(X_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
   ax = plt.subplot(2, n, i + 1)
   plt.imshow(X_test[i].reshape(28, 28))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   ax = plt.subplot(2, n, i + 1 + n)
   plt.imshow(reconstructed_img[i].reshape(28, 28))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
plt.show()
















