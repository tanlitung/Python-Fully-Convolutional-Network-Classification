import os
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from generator import Generator
from model import FCN_model
import extract_data


def train_model(model, train_generator, val_generator, epochs=5):
    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path,
                              'model_epoch_{epoch:02d}_loss_{loss:.2f}_accuracy_{accuracy:.2f}_val_loss_{val_loss:.2f}_val_accuracy_{val_accuracy:.2f}.h5')

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  callbacks=[tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                                                                save_best_only=True, verbose=1)],
                                  validation_data=val_generator,
                                  validation_steps=len(val_generator))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Accuracy/Loss')
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
    plt.show()

    return history


print("[INFO] Initializing data...")
if not os.path.exists("./dataset"):
    extract_data.extract_dataset()
    extract_data.get_dataset_statistics("./dataset")

print("[INFO] Generating data batches...")
BASE_PATH = './dataset'
train_generator = Generator(os.path.join(BASE_PATH, 'train'))
val_generator = Generator(os.path.join(BASE_PATH, 'val'))
train_image, train_label = train_generator.__getitem__(0)
plt.imshow(train_image[0], cmap=plt.cm.gray)
plt.suptitle(f"Class: {np.where(train_label[0] == 1)[0][0]}")
plt.show()

print(f"Train Image Shape: {train_image.shape}")

# Build the model
print("[INFO] Building model...")
model = FCN_model(len_classes=10)

# Clear the session before training
print("[INFO] Clearing session...")
tf.keras.backend.clear_session()
K.clear_session()

# Compile the model
print("[INFO] Compiling model...")
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Load the weights for predictions
print("[INFO] Loading weights...")
model.load_weights('./snapshots/pre-trained/epoch_05_loss_0.34_acc_0.87_val_loss_0.07_val_acc_0.98.h5')

# Train the model
print("[INFO] Training model...")
# Uncomment the line below to train the model
# train_model(model, train_generator, val_generator, epochs=5)

print("[INFO] Generating test cases...")
val_image, val_label = val_generator.__getitem__(0)
predictions = []
actual = []
for i in range(25):
    im = val_image[i]
    im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
    predictions.append(np.argmax(np.array(model.predict(im)[0])))
    actual.append(np.argmax(val_label[i]))

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
count = 0
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(val_image[count], cmap=plt.cm.gray)
        ax[i, j].set_title(f'Actual: {actual[count]} Pred: {predictions[count]}')
        ax[i, j].axis('off')
        count += 1

fig.tight_layout(pad=1.0)
plt.show()