import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model

# using pretrained yolo nano model to detect pedestrians, cars, bus, trucks.
yolo_model = YOLO("yolov8n.pt")  


path = "C:/Users/Admin/Desktop/nullclass/maybe_final_model_96%train_98%test.keras"
model = load_model(path)  

top = tk.Tk()
top.geometry('800x600')
top.title('Car Color & People Counter')
top.configure(background='#CDCDCD')

sign_image = Label(top)
label1 = Label(top, background="#CDCDCD", font=('arial', 15))
label2 = Label(top, background="#CDCDCD", font=('arial', 15))


def detect(file_path):
    try:
       
        image_pil = Image.open(file_path).convert("RGB")  
        image_np = np.array(image_pil) 

        results = yolo_model(file_path)  
        detections = results[0].boxes.data.numpy()  

        car_count = 0
        people_count = 0

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)

            if class_id in [2, 5, 7]:  # Car=2, Bus=5, Truck=5      
                ###can change id which ever is required 
                                
                car_pil = image_pil.crop((int(x1), int(y1), int(x2), int(y2)))
                car_resized = car_pil.resize((128, 128))
                car_resized = np.array(car_resized, dtype=np.float32) / 255.0
                car_resized = np.expand_dims(car_resized, axis=0)

                # Predict car color
                pred = model.predict(car_resized)[0][0]
                color_class = 1 if pred >= 0.65 else 0

                
                color = (255, 0, 0) if color_class == 1 else (0, 0, 255)   #image is in rgb format red box for blue and blue for others 
                cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                print(pred,x1,y1,x2,y2)
                car_count += 1

            elif class_id == 0:  # Person
                cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for people
                people_count += 1

       
        image_pil_final = Image.fromarray(image_np)
        im = ImageTk.PhotoImage(image_pil_final)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(foreground="#011638", text=f"Cars Detected: {car_count}")
        label2.configure(foreground="#011638", text=f"People Detected: {people_count}")

    except Exception as e:
        label1.configure(foreground="red", text=f"Error: {e}")
        label2.configure(text="")

def show_detect_button(file_path):
    detect_b = Button(top, text="Detect", command=lambda: detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        label1.configure(foreground="red", text=f"Error: {e}")
        label2.configure(text="")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = Label(top, text="Car Color & People Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
