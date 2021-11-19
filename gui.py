import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

img_width, img_height = 150, 150
# load model da save
model = load_model('model_dog_cat.h5')


# Giao diện GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Phân Loại Ảnh Chó Mèo')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 18, 'bold'),foreground='blue')
sign_image = Label(top)


def classify(path):
    img_in = image.load_img(path)
    img_rs = img_in.resize((img_width, img_height))
    input_arr = image.img_to_array(img_rs)
    input_arr = np.expand_dims(input_arr, axis=0)
    # prediction = model.predict_classes(input_arr)
    prediction = model.predict(input_arr)[0]
    print(prediction[0])
    result = ''
    if prediction[0] == 0:
        result = 'Cat'
    elif prediction[0]== 1:
        result = 'Dog'

    label.configure(foreground='green', text=result)


def show_classify_button(file_path):
    classify_b = Button(top, text="Phân Loại Ảnh", command=lambda: classify(file_path), padx=12, pady=6)
    classify_b.configure(background='#364196', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Chọn Ảnh", command=upload_image, padx=10, pady=5)
upload.configure(background='#333', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=70)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Nhận Diện Mẫu Ảnh Chó Mèo", pady=40, font=('Times New Roman', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#000')
heading.pack()
top.mainloop()