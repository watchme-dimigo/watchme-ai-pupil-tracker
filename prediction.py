from preprocessor import apply_threshold
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('./model.h5')
# load model from presaved file

def prediction(img_path, model_path='./model.h5'):

    img = image.load_img(img_path, target_size=(64, 64))
    img = np.expand_dims(image.img_to_array(img), axis=0)
    # print(img)

    result = model.predict_classes(img)
    return ['bottom_left', 'bottom_right', 'normal', 'top_left', 'top_right'][result[0]]

if __name__ == '__main__':
    res = prediction(f'./test.png')
    print(res)
    