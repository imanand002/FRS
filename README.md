# Face detection & Recognition System

## Models:

1. For face detection: Mediapipe (BlazeFace - Short Range)
2. For face recognition: FaceNet

## Dataset: Own images
-> Structure: dataset/person_name/image-1.jpeg

## Data Pipeline:
-> database.py   → builds embeddings.npy, names.npy  
-> recognise.py  → webcam recognition using those files 

