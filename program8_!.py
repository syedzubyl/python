import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
test_acc = model.evaluate(test_images, test_labels, verbose=2)[1]
print(f"Test accuracy: {test_acc:.4f}")
plt.figure(figsize=(10, 4))
for i, key in enumerate(['accuracy', 'loss']):
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[key], label='Training ' + key.capitalize())
    plt.plot(history.history['val_' + key], label='Validation ' + key.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel(key.capitalize())
    plt.legend()
plt.show()
model.save('cifar10_cnn_model.h5')
predictions = tf.keras.models.load_model('cifar10_cnn_model.h5').predict(test_images)
print(f"Predicted class: {tf.argmax(predictions[0])}")
