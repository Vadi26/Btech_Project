import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import pickle

# 1. Load the trained model from the .pkl file
def load_model(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()  # Set the model to evaluation mode
    return model

# 2. Preprocess the input image
def preprocess_image(image_path):
    # Define the same preprocessing transformations used during training
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    # Open the image and convert it to grayscale (MNIST images are grayscale)
    image = Image.open(image_path).convert('L')
    
    # Apply transformations
    image = transform(image)
    
    # Add a batch dimension (MNIST expects input shape [1, 28, 28], batch size is 1 here)
    image = image.unsqueeze(0)
    return image

# 3. Predict the digit in the image
def predict_digit(model, image_tensor):
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()  # Get the predicted digit
    return predicted_digit

# Example usage:
if __name__ == "__main__":
    # Path to your .pkl file and the input image
    pkl_file_path = "/home/vadi/oii/Academics/BtechProject/tutorial/multirun/2024-12-23/23-11-16/batch10FedAdam/results.pkl"
    image_path = "/home/vadi/oii/Academics/BtechProject/mediumArticle/archive/testSet/testSet/img_6.jpg"

    # Load the model
    model = load_model(pkl_file_path)

    # Preprocess the input image
    image_tensor = preprocess_image(image_path)

    # Get the prediction
    predicted_digit = predict_digit(model, image_tensor)

    print(f"The model predicts the digit as: {predicted_digit}")
