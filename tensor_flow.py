import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Prepare data
X_train_nn = X_train.astype(np.float32)
X_test_nn = X_test.astype(np.float32)
y_train_nn = tf.keras.utils.to_categorical(y_train, 3)
y_test_nn = tf.keras.utils.to_categorical(y_test, 3)
# Implement a simple perceptron model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
# Train the model
history = model.fit(X_train_nn, y_train_nn, epochs=50, batch_size=5,
validation_data=(X_test_nn, y_test_nn))
# Evaluate the model
loss, accuracy = model.evaluate(X_test_nn, y_test_nn)
print("Accuracy:", accuracy)
# Visualize training process
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()