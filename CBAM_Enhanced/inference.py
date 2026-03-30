import torch
import cv2
import numpy as np
from src.dsnet import DSNet_Enhanced # Assuming you move your class to src/dsnet.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def run_inference(image_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DSNet_Enhanced().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Prep Image
    transform = A.Compose([A.Resize(352, 352), A.Normalize(), ToTensorV2()])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
        mask = (output > 0.5).astype(np.uint8) * 255

    cv2.imwrite("prediction.png", mask)
    print("✅ Prediction saved as prediction.png")

if __name__ == "__main__":
    run_inference("test_image.jpg", "weights/DSNet_Enhanced_Best.pth")