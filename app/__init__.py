
import torch
import cv2
import sqlite3
import pathlib

# Fix for PosixPath error on Windows
pathlib.PosixPath = pathlib.WindowsPath

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # create a table if it doesn't exist
    cursor.execute(''' 
                   CREATE TABLE IF NOT EXISTS violations(
                    ID INTEGER PRIMARY KEY AUTOINCREMENT, 
                    bike_id TEXT NOT NULL, 
                    riders INTEGER NOT NULL, 
                   no_helmet INTEGER , 
                   image TEXT
                   ) ''')
    

    cursor.execute('''  
                    CREATE TABLE IF NOT EXISTS messages(
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    message TEXT NOT NULL
                    )
                   ''')

    conn.commit()
    conn.close()

# initialize the database
init_db()

model = torch.hub.load("ultralytics/yolov5", "custom", path="app/model/project_model_kaggle.pt")

def my_model():
    return model