# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,
Flatten, Dense
from tensorflow.keras.utils import to_categorical
# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape and normalize data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), activation=&#39;relu&#39;,
input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation=&#39;relu&#39;))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation=&#39;relu&#39;))
model.add(Dense(10, activation=&#39;softmax&#39;))
# Compile model
model.compile(optimizer=&#39;adam&#39;,
loss=&#39;categorical_crossentropy&#39;,
metrics=[&#39;accuracy&#39;])
# Train model
history = model.fit(X_train, y_train,
epochs=5,
batch_size=128,
validation_split=0.1)
# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(&quot;Test Accuracy:&quot;, test_accuracy)
# Plot accuracy graph
plt.plot(history.history[&#39;accuracy&#39;], label=&#39;Training Accuracy&#39;)
plt.plot(history.history[&#39;val_accuracy&#39;], label=&#39;Validation
Accuracy&#39;)
plt.xlabel(&#39;Epochs&#39;)
plt.ylabel(&#39;Accuracy&#39;)
plt.legend()
plt.title(&#39;CNN Accuracy Curve&#39;)
plt.show()
