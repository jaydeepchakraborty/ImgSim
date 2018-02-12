
from ImgConst import *

def prepare_data():

    os.chdir(IMG_DIR)
    
    image_arr = []
    for file in glob.glob(FILE_EXT):
        img = cv2.imread(IMG_DIR + file)
        img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
        out = []
        for x in range(IMG_WIDTH):
            for y in range(IMG_HEIGHT):
                for z in range(IMG_COLOR):
                    out.append(img[x][y][z])
                                        
        image_arr.append(out)
        
    img_vec_arr = np.array([np.asarray(image) for image in image_arr])
    
    return img_vec_arr

def img_vec(img_file_nm):

    os.chdir(IMG_DIR)
    
    image_arr = []
    img = cv2.imread(IMG_DIR + img_file_nm)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    out = []
    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            for z in range(IMG_COLOR):
                out.append(img[x][y][z])
                                    
    image_arr.append(out)
        
    img_vec_arr = np.array([np.asarray(image) for image in image_arr])
    
    return img_vec_arr
