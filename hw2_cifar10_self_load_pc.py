import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_batch(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X = data_dict[b'data']
        Y = data_dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3)
        Y = np.array(Y)
        return X, Y

def load_all_batches(folder_path):
    x_list, y_list = [], []
    for i in range(1, 6):
        X, Y = load_batch(f"{folder_path}/data_batch_{i}")
        x_list.append(X)
        y_list.append(Y)
    x_train = np.concatenate(x_list)
    y_train = np.concatenate(y_list)
    x_test, y_test = load_batch(f"{folder_path}/test_batch")
    return x_train, y_train, x_test, y_test

# 資料路徑
folder = "C:/Users/user/Downloads/cifar-10-python/cifar-10-batches-py"
x_train, y_train, x_test, y_test = load_all_batches(folder)

# 資料型態轉換與正規化 (0~1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 標籤 one-hot 編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 建構 CNN 模型
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Dropout(0.3),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.3),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.3),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 訓練模型
epochs = 200
batch_size = 500

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# 評估 train/test loss 與 accuracy
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"Train Loss: {train_loss:.4f}  |  Train Accuracy: {train_acc:.4f}")
print(f"Test Loss:  {test_loss:.4f}  |  Test Accuracy:  {test_acc:.4f}")

# 繪製訓練歷程圖 (Loss & Accuracy)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
