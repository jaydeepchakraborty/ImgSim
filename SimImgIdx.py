from ImgConst import *

def get_sorted_similarity_idx(encoder, img_to_find, encoded_images, loss):
    

    #initializing vars to pass into tensorflow
    encoded_images = encoder.predict(img_to_find)
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)

    X_selected = encoded_images.tolist()
    X_all = encoded_images.tolist()

    X_selected_tf = tf.Variable(X_selected, tf.float32)
    X_all_tf = tf.Variable(X_all, tf.float32)
    if loss=='binary_crossentropy':
        loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
    elif loss=='cosine_proximity':
        loss_tf = cosine_proximity(X_selected_tf, X_all_tf)
    else:
        print('Unknown loss, using binary_crossentropy.')
        loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
    init_op = tf.global_variables_initializer()

    similarity = []
    with tf.Session() as sess:
        sess.run(init_op)
        similarity = sess.run(loss_tf)
    similarity_sorted = np.argsort(np.array(similarity)) #the same figure appears in X_all too, so remove it
    
    return similarity_sorted