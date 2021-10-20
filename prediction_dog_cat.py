from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import io

#kich thuoc anh
img_width, img_height = 150, 150
# load model da save
model = load_model('model_dog_cat.h5')
my_path = "predict/"
files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
print(files)

#dem so luong
dog_counter = 0
cat_counter  = 0

for file in files :
    img_in = image.load_img(my_path+file)
    #img = Image.open(io.BytesIO(img_in))
    #img = img.resize((img_width, img_height))
    img_rs = img_in.resize((img_width,img_height))
    input_arr = image.img_to_array(img_rs)
    input_arr = np.expand_dims(input_arr,axis=0)
    prediction = model.predict_classes(input_arr)
    if prediction[0][0] == 0:
        print(file," : cat")
        cat_counter += 1
    elif prediction[0][0] == 1:
        print(file," : dog")
        dog_counter += 1
print("total dogs :  ",dog_counter)
print("total cats :  ",cat_counter)