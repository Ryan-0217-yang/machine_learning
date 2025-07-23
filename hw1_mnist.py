import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD
from sklearn.metrics import classification_report

# 設定參數
epochs = 20
batch_size = 50
row_col = 48  # VGG16 輸入尺寸

# 載入並處理 MNIST 資料
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, y_train = X_train[:5000], y_train[:5000]
    X_test, y_test = X_test[5000:6000], y_test[5000:6000]

    X_train = [cv2.cvtColor(cv2.resize(i, (row_col, row_col)), cv2.COLOR_GRAY2RGB) for i in X_train]
    X_train = np.stack(X_train).astype('float32')

    X_test = [cv2.cvtColor(cv2.resize(i, (row_col, row_col)), cv2.COLOR_GRAY2RGB) for i in X_test]
    X_test = np.stack(X_test).astype('float32')

    X_train /= 255
    X_test /= 255

    y_train_ohe = to_categorical(y_train, 10)
    y_test_ohe = to_categorical(y_test, 10)

    return (X_train, y_train_ohe, y_train), (X_test, y_test_ohe, y_test)

# 建立模型
def load_model():
    base_network = VGG16(include_top=False, weights='imagenet', input_shape=(row_col, row_col, 3))
    for layer in base_network.layers:
        layer.trainable = False

    model = Flatten()(base_network.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dense(4096, activation='relu', name='fc3')(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax', name='predictions')(model)

    return Model(base_network.input, model, name='vgg16_mnist')

# 載入模型與資料
model = load_model()
(x_train, y_train_ohe, y_train), (x_test, y_test_ohe, y_test) = load_data()

# 編譯模型
sgd = SGD(learning_rate=0.05, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 訓練模型
history = model.fit(
    x_train, y_train_ohe,
    validation_data=(x_test, y_test_ohe),
    epochs=epochs,
    batch_size=batch_size
)

# 評估結果
train_acc = model.evaluate(x_train, y_train_ohe, verbose=0)[1]
test_acc = model.evaluate(x_test, y_test_ohe, verbose=0)[1]
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# 預測並輸出每個數字的 precision / recall / f1-score / support
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 繪製 Accuracy 曲線
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# 繪製 Loss 曲線
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
