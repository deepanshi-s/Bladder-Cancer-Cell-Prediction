import tkinter
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import ntpath
import os
import subprocess
import time

head1="unet_final"                     #add location of code folder
def select_image():
        global path 
        path = filedialog.askopenfilename()
        run()
        ogimage(path)

        
def ogimage(path):
        #edged1(path)
        edge_it(path)
        global panelA, panelB,panelD
        global image
        image = cv2.imread(path)
        image = cv2.resize(image,(500,500))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        if panelA is None or panelD is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            #panelB = Label(image=edged)
            #panelB.image = edged
            #panelB.pack(side="top", padx=10, pady=10)
            panelD = Label(image=img1)
            panelD.image = img1
            panelD.pack(side="bottom", padx=10, pady=10)

        else:
            panelA.configure(image=image)
            #panelB.configure(image=edged)
            panelA.image = image
            #panelB.image = edged
            panelD.configure(image=img1)
            panelD.image = img1           
            
        

def edge_it(pt):
    global img1
    head, tail = ntpath.split(pt)
    path5 = head1 + "\\pred_" + tail

    img1 = cv2.imread(path5)
    img1 = cv2.resize(img1,(500,500))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    img1 = ImageTk.PhotoImage(img1)

    
def run():
        args = ["predict.py"]
        args.extend("--image".split())
        args.extend(path.split())
        args.extend("--checkpoint_path checkpoints/latest_model_MobileUNet_CamVid.ckpt --model MobileUNet".split())   #add path of the checkpoints
        subprocess.Popen(['python'] + args)
        time.sleep(25)

root = Tk()
panelA = None
panelB = None
panelD = None
btn = Button(root, text="Select an image", command=select_image,width=20)
btn.pack()
panelC=Label(root,text="Metrics here",font=40)
panelC.pack(side="left")
fo = open("out.txt", "r")
line = fo.read()
panelC.configure(text=line)
root.mainloop()
