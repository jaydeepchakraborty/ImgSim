from ImgConst import *

def get_model():
    
    input_img = Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_COLOR))  
    x = Conv2D(16, (3, 3), activation=ACTV_FUNC_1, padding=PAD)(input_img)
    x = MaxPooling2D((2, 2), padding=PAD)(x)
    x = Conv2D(8, (3, 3), activation=ACTV_FUNC_1, padding=PAD)(x)
    x = MaxPooling2D((2, 2), padding=PAD)(x)
    x = Conv2D(8, (3, 3), activation=ACTV_FUNC_1, padding=PAD)(x)
    encoded = MaxPooling2D((2, 2), padding=PAD, name=ENC_NM)(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation=ACTV_FUNC_1, padding=PAD)(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation=ACTV_FUNC_1, padding=PAD)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation=ACTV_FUNC_1)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation=ACTV_FUNC_2, padding=PAD)(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=OPT_1, loss=LOSS_1)
    
    return autoencoder


def train_model(x_train, autoencoder):
    
    x_train = x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), IMG_WIDTH, IMG_HEIGHT, IMG_COLOR))
    
    
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    num_trng_batch_per_epoch = math.ceil(x_train.shape[0]/TRNG_BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        print('epoch: {}'.format(epoch))
        for batchnum in range(num_trng_batch_per_epoch):
            x_train_batch = x_train[batchnum*TRNG_BATCH_SIZE : (batchnum+1)*TRNG_BATCH_SIZE]
            x_train_noisy_batch = x_train_noisy[batchnum*TRNG_BATCH_SIZE : (batchnum+1)*TRNG_BATCH_SIZE]
     
            autoencoder.fit(x_train, x_train,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                verbose=0,
                                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
    
    return autoencoder

def save_model(autoencoder):
    file_path = MODEL_FL_PATH.format(int(round(time.time() * 1000)))
    autoencoder.save(file_path)
    
    return file_path
    
    