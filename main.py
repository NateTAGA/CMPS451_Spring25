import numpy as np
from sklearn.datasets import fetch_openml
from hmmlearn import hmm
import joblib

# --- Load MNIST Dataset ---
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# Normalize and reshape images
X = X / 255.0
images = X.reshape(-1, 28, 28)

# --- Feature Extraction ---
def extract_features(image, num_slices=14):
    h, w = image.shape
    slice_width = w // num_slices
    features = []
    for i in range(num_slices):
        slice_img = image[:, i*slice_width:(i+1)*slice_width]
        density = np.sum(slice_img)
        center_of_mass = np.mean(np.argwhere(slice_img)) if np.any(slice_img) else 0
        features.append([density, center_of_mass])
    return np.array(features)

# --- Train HMMs for Digits 0–9 ---
def train_models():
    models = {}
    print("Training HMM models (this may take a minute)...")
    for digit in range(10):
        digit_imgs = images[y == digit][:100]  # Limit to 100 samples per digit
        sequences = [extract_features(img) for img in digit_imgs]
        lengths = [len(seq) for seq in sequences]
        X_concat = np.vstack(sequences)

        model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100)
        model.fit(X_concat, lengths)
        models[digit] = model
        print(f"Trained HMM for digit {digit}")
        joblib.dump(model, f"hmm_model_{digit}.pkl")  # Save model
    return models

# --- Predict One Sample ---
def predict_sample(models, test_img, true_label):
    features = extract_features(test_img)
    scores = {digit: model.score(features) for digit, model in models.items()}
    predicted = max(scores, key=scores.get)
    print("\n--- Prediction Result ---")
    print(f"Actual digit: {true_label}")
    print(f"Predicted digit: {predicted}")

# --- Main Execution ---
if __name__ == "__main__":
    models = train_models()
    test_img = images[0]
    test_label = y[0]
    predict_sample(models, test_img, test_label)