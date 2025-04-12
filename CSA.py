import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import torch
from torchvision import transforms

def load_gei_dataset(root_dir):
    """
    Load GEI dataset
    :param root_dir: Root directory of the dataset (GEI)
    :return: Image tensors (batch_size, height, width), label list
    """
    images = []
    labels = []
    
    # Iterate through each person's folder
    for person_id in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_id)
        if not os.path.isdir(person_dir):
            continue
            
        # Iterate through each feature-specific folder (bg-01, bg-02, ...)
        for gait_instance in os.listdir(person_dir):
            instance_dir = os.path.join(person_dir, gait_instance)
            if not os.path.isdir(instance_dir):
                continue
                
            # Iterate through each angle folder (000, 018, ..., 180)
            for angle in os.listdir(instance_dir):
                angle_dir = os.path.join(instance_dir, angle)
                if not os.path.isdir(angle_dir):
                    continue
                
                # Construct image path
                image_name = f"{angle}_GEI.png"  # Image name format
                image_path = os.path.join(angle_dir, image_name)
                    
                # Check if the image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} does not exist, skipping")
                    continue
                    
                # Open the image (already grayscale)
                image = Image.open(image_path)
                
                # Convert to PyTorch tensor
                transform = transforms.ToTensor()
                image_tensor = transform(image).squeeze(0)  # Remove channel dimension
                    
                # Add to lists
                images.append(image_tensor)
                labels.append(int(person_id))  # Use folder name as label
    
    # Convert lists to tensors
    images = torch.stack(images)  # (batch_size, height, width)
    labels = torch.tensor(labels)  # (batch_size,)
    
    return images, labels


def csa(X, k, l, max_iter=10, tol=1e-6):
    """
    Coupled Subspace Analysis (CSA)
    :param X: Input data (batch_size, height, width)
    :param k: Target reduced dimension for U
    :param l: Target reduced dimension for V
    :param max_iter: Maximum number of iterations
    :param tol: Convergence threshold
    :return: Low-dimensional representation Z, and projection matrices U, V
    """
    batch_size, m, n = X.shape
    
    # Initialize U and V, either randomly or with PCA initialization
    U = torch.randn(m, k, dtype=torch.float32, requires_grad=False)
    V = torch.randn(n, l, dtype=torch.float32, requires_grad=False)
    
    prev_loss = float('inf')
    for iteration in range(max_iter):
        # Fix V, update U
        XV = torch.matmul(X, V)  # (batch_size, m, l)
        XV_2d = XV.reshape(-1, m)  # Convert to 2D matrix
        U, _ = torch.linalg.qr(XV_2d.T)  # Compute orthogonal basis
        U = U[:, :k]  # Take the first k dimensions
        
        # Fix U, update V
        UX = torch.matmul(U.T, X)  # (k, batch_size, n)
        UX_2d = UX.reshape(-1, n)  # Convert to 2D matrix
        V, _ = torch.linalg.qr(UX_2d.T)  # Compute orthogonal basis
        V = V[:, :l]  # Take the first l dimensions
        
        # Calculate loss
        Z = torch.matmul(U.T, X).matmul(V)  # Reduced data
        loss = torch.norm(X - torch.matmul(U, Z).matmul(V.T), p='fro') / batch_size
        
        print(f"Iteration {iteration+1}, Loss: {loss.item()}")
        
        # Check convergence
        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()
    
    return Z, U, V

# Example usage
if __name__ == "__main__":
    # Dataset root directory
    root_dir = r"C:\Users\89778\Desktop\Project\GaitDatasetB-silh\GEI"

    # Load data
    images, labels = load_gei_dataset(root_dir)

    # Print dataset info
    print("Image tensor shape:", images.shape)  # (batch_size, 64, 64)
    print("Label tensor shape:", labels.shape)  # (batch_size,)
    images = images / 255.0  # Normalize to [0, 1]
    mean = images.mean(dim=(1, 2), keepdim=True)  # Calculate mean for each image
    images = images - mean  # Center the data
    
    # Run CSA
    k, l = 50, 40  # Target reduced dimensions
    Z, U, V = csa(images, k, l)

    # Print results
    print("Low-dimensional representation Z shape:", Z.shape)  # (batch_size, k, l)
    print("Projection matrix U shape:", U.shape)  # (height, k)
    print("Projection matrix V shape:", V.shape)  # (width, l)