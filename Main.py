from ImgConst import *

from LoadData import prepare_data, img_vec
from Model import get_model, train_model, save_model
from SimImgIdx import get_sorted_similarity_idx
from ImgFeatures import save_img_feature, load_img_features

try:
    x_train = prepare_data()
    autoencoder = get_model()
    autoencoder = train_model(x_train,autoencoder)
    save_model(autoencoder) 
    autoencoder_save = load_model(MODEL_FL_PATH)
    save_img_feature(autoencoder_save, x_train)
    feature_val = load_img_features()


    img_test_file = "f0002_05.png"
    img_test = img_vec(img_test_file)
    img_test = img_test.astype('float32') / 255.
    img_test = np.reshape(img_test, (len(img_test),IMG_WIDTH, IMG_HEIGHT, IMG_COLOR))
    similarity_sorted = get_sorted_similarity_idx(autoencoder_save, img_test, encoded_images=feature_val, loss=LOSS_2)
    similar_idx = similarity_sorted[0]
    
    print(similar_idx)
    print(similarity_sorted)
    
#     f = plt.figure(figsize=(20, 4))
#     f.add_subplot(1, 2, 1)
#     plt.imshow(x_train[1].reshape(28, 28))
#     f.add_subplot(1,2, 2)
#     plt.imshow(x_train[similar_idx].reshape(28, 28))
#     plt.show(block=True)
    
    
except Exception as e:
    print(e)


