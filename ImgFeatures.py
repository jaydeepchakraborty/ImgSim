from ImgConst import *

def save_img_feature(encoder, dataset):
    dataset = dataset.astype('float32') / 255.
    dataset = np.reshape(dataset, (len(dataset), IMG_WIDTH, IMG_HEIGHT, IMG_COLOR))
    
    encoded_images = encoder.predict(dataset)
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)
    
    np.savetxt(SAVE_FEATURE_PATH, encoded_images)
    
def load_img_features():
    return np.loadtxt(SAVE_FEATURE_PATH)
        
    