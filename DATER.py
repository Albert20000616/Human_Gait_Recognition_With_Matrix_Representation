import torch
import numpy as np
from scipy.linalg import eigh
import os
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load GEI Dataset
def load_gei_dataset(root_dir):
    """
    Load GEI dataset
    :param root_dir: Root directory of the dataset (GEI)
    :return: Image tensors (batch_size, height, width), label list
    """
    images = []
    labels = []

    for person_id in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_id)
        if not os.path.isdir(person_dir):
            continue

        for gait_instance in os.listdir(person_dir):
            instance_dir = os.path.join(person_dir, gait_instance)
            if not os.path.isdir(instance_dir):
                continue

            for angle in os.listdir(instance_dir):
                angle_dir = os.path.join(instance_dir, angle)
                if not os.path.isdir(angle_dir):
                    continue

                image_name = f"{angle}_GEI.png"
                image_path = os.path.join(angle_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} does not exist, skipping")
                    continue

                image = Image.open(image_path)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),  # Ensure consistent size
                    transforms.ToTensor()
                ])
                image_tensor = transform(image).squeeze(0)  # Remove channel dimension

                images.append(image_tensor)
                labels.append(int(person_id))

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


# 2. DATER Dimensionality Reduction Method (keep unchanged, record accuracy per iteration)
class DATER:
    def __init__(self, k, l, max_iter=100, epsilon=1e-6):
        """
        Initialize DATER algorithm
        :param k: Target reduced dimension for U
        :param l: Target reduced dimension for V
        :param max_iter: Maximum iterations
        :param epsilon: Convergence threshold
        """
        self.k = k
        self.l = l
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.U = None
        self.V = None
        self.is_fitted = False
        self.rank1_scores = []
        self.rank5_scores = []

    def fit(self, X, labels):
        """
        Train DATER model while recording accuracy per iteration
        :param X: Input matrix list (m x n)
        :param labels: Class labels
        """
        m, n = X[0].shape
        N = len(X)

        # Ensure k and l don't exceed input dimensions
        self.k = min(self.k, m)
        self.l = min(self.l, n)

        # Initialize projection matrices
        self.U = np.eye(m, self.k)
        self.V = np.eye(n, self.l)

        classes = np.unique(labels)
        for iteration in range(self.max_iter):
            # Step 1: Project to U
            X_u = [self.U.T @ x_i for x_i in X]
            X_c_u = []
            for c in classes:
                class_samples = [X_u[i] for i in range(N) if labels[i] == c]
                X_c_u.append(np.mean(class_samples, axis=0))
            X_mean_u = np.mean(X_u, axis=0)

            # Calculate scatter matrices
            S_b_u = np.zeros((n, n), dtype=np.float64)
            S_w_u = np.zeros((n, n), dtype=np.float64)

            for c_idx, c in enumerate(classes):
                X_c_u_diff = X_c_u[c_idx] - X_mean_u
                class_count = np.sum(labels == c)
                S_b_u += class_count * np.real(X_c_u_diff.T @ X_c_u_diff)

            for i in range(N):
                c_idx = np.where(classes == labels[i])[0][0]
                X_i_u_diff = X_u[i] - X_c_u[c_idx]
                S_w_u += np.real(X_i_u_diff.T @ X_i_u_diff)

            S_w_u += np.eye(n) * 1e-10  # Numerical stability

            # Update V
            eigenvalues, eigenvectors = eigh(S_b_u, S_w_u)
            V_new = eigenvectors[:, ::-1][:, :self.l]

            # Step 2: Project to V
            X_v = [x_i @ V_new for x_i in X]
            X_c_v = []
            for c in classes:
                class_samples = [X_v[i] for i in range(N) if labels[i] == c]
                X_c_v.append(np.mean(class_samples, axis=0))
            X_mean_v = np.mean(X_v, axis=0)

            # Calculate scatter matrices
            S_b_v = np.zeros((m, m), dtype=np.float64)
            S_w_v = np.zeros((m, m), dtype=np.float64)

            for c_idx, c in enumerate(classes):
                X_c_v_diff = X_c_v[c_idx] - X_mean_v
                class_count = np.sum(labels == c)
                S_b_v += class_count * np.real(X_c_v_diff @ X_c_v_diff.T)

            for i in range(N):
                c_idx = np.where(classes == labels[i])[0][0]
                X_i_v_diff = X_v[i] - X_c_v[c_idx]
                S_w_v += np.real(X_i_v_diff @ X_i_v_diff.T)

            S_w_v += np.eye(m) * 1e-10  # Numerical stability

            # Update U
            eigenvalues, eigenvectors = eigh(S_b_v, S_w_v)
            U_new = eigenvectors[:, ::-1][:, :self.k]

            # Evaluate current iteration's performance
            Z_iter = [U_new.T @ x @ V_new for x in X]
            rank1_acc, rank5_acc = evaluate(Z_iter, labels)
            self.rank1_scores.append(rank1_acc)
            self.rank5_scores.append(rank5_acc)
            print(f"DATER Iteration {iteration + 1}, Rank-1: {rank1_acc:.4f}, Rank-5: {rank5_acc:.4f}")

            # Check convergence
            if (np.linalg.norm(U_new - self.U) < m * self.epsilon and
                    np.linalg.norm(V_new - self.V) < n * self.epsilon):
                print(f"DATER Converged at iteration {iteration + 1}")
                break

            self.U = U_new
            self.V = V_new

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform data using projection matrices
        :param X: Input matrix list
        :return: Transformed matrix list
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return [self.U.T @ x @ self.V for x in X]

    def fit_transform(self, X, labels):
        """
        Train and transform data
        """
        return self.fit(X, labels).transform(X)


# 3. Classification and Evaluation
def evaluate(Z, labels, train_size_ratio=0.2):
    """
    Use nearest neighbor classifier for prediction and calculate Rank-1 & Rank-5 accuracy
    :param Z: Low-dimensional representations (list of matrices)
    :param labels: Class labels
    :param train_size_ratio: Training set ratio
    :return: Rank-1 and Rank-5 accuracy
    """
    Z_flat = np.array([z.flatten() for z in Z])
    labels = np.array(labels)

    train_size = int(len(Z) * train_size_ratio)
    Z_train, Z_test, labels_train, labels_test = train_test_split(
        Z_flat, labels, train_size=train_size_ratio, random_state=None, stratify=labels
    )

    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(Z_train)
    distances, indices = nbrs.kneighbors(Z_test)

    rank1_correct = 0
    rank5_correct = 0
    for i in range(len(Z_test)):
        if labels_train[indices[i, 0]] == labels_test[i]:
            rank1_correct += 1
        if labels_test[i] in labels_train[indices[i, :5]]:
            rank5_correct += 1

    rank1_acc = rank1_correct / len(Z_test)
    rank5_acc = rank5_correct / len(Z_test)
    return rank1_acc, rank5_acc


# 4. Visualization (mimic Fig. 3 and Fig. 4 from paper)
def plot_iteration_performance(dater_rank1_scores, dater_rank5_scores):
    """
    Plot Rank-1 and Rank-5 accuracy vs iteration (mimic Fig. 3)
    :param dater_rank1_scores: DATER's Rank-1 scores per iteration
    :param dater_rank5_scores: DATER's Rank-5 scores per iteration
    """
    plt.figure(figsize=(6, 4))
    iterations = range(1, len(dater_rank1_scores) + 1)

    plt.plot(iterations, np.array(dater_rank1_scores) * 100, label='Rank-1', marker='o')
    plt.plot(iterations, np.array(dater_rank5_scores) * 100, label='Rank-5', marker='s')
    plt.xlabel('Iteration Number')
    plt.ylabel('Recognition Accuracy (%)')
    plt.title('Rank-1 and Rank-5 Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dimension_performance(k_values, l_values, rank1_scores, rank5_scores):
    """
    Plot Rank-1 and Rank-5 accuracy vs reduced dimensions (mimic Fig. 4)
    :param k_values: List of k values for DATER
    :param l_values: List of l values for DATER
    :param rank1_scores: Rank-1 accuracy matrix (len(l_values), len(k_values))
    :param rank5_scores: Rank-5 accuracy matrix (len(l_values), len(k_values))
    """
    # Calculate k * l as x-axis
    kl_values = [k * l for k in k_values for l in l_values]
    rank1_flat = []
    rank5_flat = []

    # Flatten rank1_scores and rank5_scores
    for i in range(len(l_values)):
        for j in range(len(k_values)):
            rank1_flat.append(rank1_scores[i, j])
            rank5_flat.append(rank5_scores[i, j])

    # Sort by kl_values
    sorted_indices = np.argsort(kl_values)
    kl_values = np.array(kl_values)[sorted_indices]
    rank1_flat = np.array(rank1_flat)[sorted_indices] * 100  # Convert to percentage
    rank5_flat = np.array(rank5_flat)[sorted_indices] * 100  # Convert to percentage

    # Create subplots
    plt.figure(figsize=(12, 5))

    # Rank-1
    plt.subplot(1, 2, 1)
    plt.plot(kl_values, rank1_flat, marker='o')
    plt.xlabel('Dimension (k × l)')
    plt.ylabel('Rank-1 Accuracy (%)')
    plt.title('Rank-1 Performance')
    plt.grid(True)

    # Rank-5
    plt.subplot(1, 2, 2)
    plt.plot(kl_values, rank5_flat, marker='s')
    plt.xlabel('Dimension (k × l)')
    plt.ylabel('Rank-5 Accuracy (%)')
    plt.title('Rank-5 Performance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 5. Main Program (DATER only)
if __name__ == "__main__":
    # Load data
    root_dir = "./GEI"
    images, labels = load_gei_dataset(root_dir)
    print("Image tensor shape:", images.shape)
    print("Label tensor shape:", labels.shape)

    # Preprocessing
    images = images / 255.0  # Normalize
    mean = images.mean(dim=(1, 2), keepdim=True)
    images = images - mean  # Center data

    # Convert to NumPy format for DATER
    images_np = images.numpy()
    X = [images_np[i] for i in range(images_np.shape[0])]
    labels_np = labels.numpy()

    # Parameter settings
    k_values = [20, 30, 40, 50]  # DATER's k values (≤ original dimension 64)
    l_values = [20, 30, 40]  # DATER's l values (≤ original dimension 64)

    # 1. Plot performance vs iterations (similar to Fig. 3)
    print("Evaluating performance vs iterations...")
    dater = DATER(k=k_values[0], l=l_values[0], max_iter=5)
    Z_dater = dater.fit_transform(X, labels_np)
    plot_iteration_performance(dater.rank1_scores, dater.rank5_scores)

    # 2. Plot performance vs reduced dimensions (similar to Fig. 4)
    print("Evaluating performance vs reduced dimensions...")
    rank1_scores = np.zeros((len(l_values), len(k_values)))
    rank5_scores = np.zeros((len(l_values), len(k_values)))
    for i, l_dater in enumerate(l_values):
        for j, k_dater in enumerate(k_values):
            print(f"Running DATER: k_dater={k_dater}, l_dater={l_dater}")
            dater = DATER(k=k_dater, l=l_dater, max_iter=5)
            Z_dater = dater.fit_transform(X, labels_np)
            rank1_acc, rank5_acc = evaluate(Z_dater, labels_np)
            rank1_scores[i, j] = rank1_acc
            rank5_scores[i, j] = rank5_acc
            print(f"Rank-1 Accuracy: {rank1_acc:.4f}, Rank-5 Accuracy: {rank5_acc:.4f}")

    plot_dimension_performance(k_values, l_values, rank1_scores, rank5_scores)

    # Output best results
    best_idx = np.argmax(rank1_scores)
    best_l_idx, best_k_idx = np.unravel_index(best_idx, rank1_scores.shape)
    best_k = k_values[best_k_idx]
    best_l = l_values[best_l_idx]
    print(f"\nBest parameters: k={best_k}, l={best_l}")
    print(f"Best Rank-1 Accuracy: {rank1_scores[best_l_idx, best_k_idx]:.4f}")
    print(f"Best Rank-5 Accuracy: {rank5_scores[best_l_idx, best_k_idx]:.4f}")
