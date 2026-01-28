import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FaceNetRecognizer:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def get_embedding(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)

        tensor = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.cpu().numpy()[0]
