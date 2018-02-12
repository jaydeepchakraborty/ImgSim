from ImgConst import *

def save_img_feature(encoder, dataset):
    dataset = dataset.astype('float32') / 255.
    dataset = np.reshape(dataset, (len(dataset), IMG_WIDTH, IMG_HEIGHT, IMG_COLOR))
    
    encoded_images = encoder.predict(dataset)
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)
    
    feature_path = SAVE_FEATURE_PATH.format(int(round(time.time() * 1000)))
    np.savetxt(feature_path, encoded_images)
    
    return feature_path
    
def load_img_features(feature_path):
    return np.loadtxt(feature_path)
        
    