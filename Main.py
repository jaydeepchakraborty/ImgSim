from LoadData import get_data
from Model import get_model
from keras.callbacks import TensorBoard
import numpy as np
import math
import time


(x_train, x_test) = get_data()

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



EPOCHS = 20
BATCH_SIZE = 64
TRNG_BATCH_SIZE = 1000
num_trng_batch_per_epoch = math.ceil(x_train.shape[0]/TRNG_BATCH_SIZE)


autoencoder = get_model()

for epoch in range(EPOCHS):
    print('epoch: {}'.format(epoch))
    for batchnum in range(num_trng_batch_per_epoch):
        x_train_batch = x_train[batchnum*TRNG_BATCH_SIZE : (batchnum+1)*TRNG_BATCH_SIZE]
        x_train_noisy_batch = x_train_noisy[batchnum*TRNG_BATCH_SIZE : (batchnum+1)*TRNG_BATCH_SIZE]

        autoencoder.fit(x_train_noisy, x_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            validation_data=(x_test_noisy, x_test),
                            callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

autoencoder.save('autoencoder_{}.h5'.format(int(round(time.time() * 1000))))


