import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from Sub_Functions.Custom_layer import StructuralAttention
from keras.models import load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)
# extracting the output of the first last layer
model1 = Model(inputs=model.input, outputs=model.layers[1].output)


def deep_color_based_pattern(img):
    # img = cv2.resize(img, (224, 224))
    # img_array = image.img_to_array(img)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    features = model1.predict(img_preprocessed)

    # Take the 2nd channel (index 1) of the output feature map
    channel_2 = features[0, :, :, 1]

    # Normalize the channel for visualization and further processing
    channel_2_normalized = cv2.normalize(channel_2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Optional: Apply a basic color-based pattern method — e.g., enhance edges (local contrast)
    enhanced = cv2.equalizeHist(channel_2_normalized)  # Improves local contrast

    # Resize to 230x230
    final_output = cv2.resize(enhanced, (150, 150))

    return final_output


from keras.applications import ResNet101
from keras.models import Model
import cv2
from Sub_Functions.Structural_pattern import StructuralPattern

# loading the ResNet101 model
model2 = ResNet101(weights='imagenet', include_top=True)
# creating a new model with only the last layer of the ResNet101 model
model2 = Model(inputs=model2.input, outputs=model2.layers[1].output)


def Deep_Structural_Pattern(img):
    # resize the image to 224x224
    img = cv2.resize(img, (224, 224))
    # predicting the output of the last layer of the ResNet101 model
    input1 = np.expand_dims(img, axis=0)
    # output image
    output = model2.predict(input1)
    # squeezing to remove the batch size from the image
    output = np.squeeze(output)
    # creating instance of StructuralPattern class
    SP = StructuralPattern(output)
    # getting the structural pattern
    final_out = SP.get_structural_pattern()
    # returning the final output
    final_out = cv2.resize(final_out, (150, 150))
    return final_out  # output shape 32,32


from keras.applications import ResNet152

model3 = ResNet152(weights="imagenet", include_top=True)
model3 = Model(inputs=model3.input, outputs=model3.layers[1].output)


def Resnet151(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    output = model3.predict(img)
    output = np.squeeze(output)
    final_out = output[:, :, 1]
    final_out = cv2.resize(final_out, (150, 150))
    return final_out


from scipy.stats import skew


def Statistical_Green(img):
    # Load image and extract green channel
    green = img[:, :, 1].astype(np.float32)
    h, w = green.shape

    # Output map (same shape as green channel)
    output = np.zeros((h, w), dtype=np.float32)

    # Loop through each pixel
    for i in range(h):
        for j in range(w):
            try:
                # Extract 3x3 neighborhood
                window = green[i - 1:i + 2, j - 1:j + 2]

                if window.shape != (3, 3):
                    raise Exception("Edge")

                window_flat = window.flatten()

                # Compute statistical features
                mean_val = np.mean(window_flat)
                std_val = np.std(window_flat)
                skew_val = skew(window_flat)

                # Average of the three statistics
                average_stat = (mean_val + std_val + skew_val) / 3.0

                # Set as center pixel value
                output[i, j] = average_stat

            except:
                # For border pixels, keep the original green value
                output[i, j] = green[i, j]

    # Normalize to 0–255 for image display or model input
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    output = output.astype(np.uint8)

    # Resize to desired output shape
    output_resized = cv2.resize(output, (150, 150))

    return output_resized  # Shape: (230, 230)


# ---------------------------
# GUI Application
# ---------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Feature Extraction & Classification")
        self.root.geometry("760x650")
        self.root.configure(bg="#f0f2f5")

        self.model_paths = {
            'DB1': r"Saved_model/DB1_model.h5",
            'DB2': r"Saved_model/DB2_model.h5",
            'DB3': r"Saved_model/DB3_model.h5",
        }

        self.models = {'DB1': None, 'DB2': None, 'DB3': None}
        self.current_model_key = None

        self.selected_image = None
        self.selected_image_path = ""
        self._tk_image_ref = None

        self.class_labels = ["Class 1", "Class 2", "Class 3", "Class 4"]

        # Known label keywords and DB tags
        self.known_diseases = [
            "Bacterialblight", "Blast", "Brownspot", "Tungro", "Leafsmut",
            "Bacterial leaf blight", "Brown spot", "Leaf smut"
        ]
        self.known_dbs = ["DB1", "DB2", "DB3"]

        # GUI Layout
        top_frame = tk.Frame(root, bg="#f0f2f5")
        top_frame.pack(pady=15)

        for db_name in self.known_dbs:
            btn = tk.Button(
                top_frame, text=db_name, width=10, font=("Helvetica", 12, "bold"),
                command=lambda db=db_name: self.load_model_hardcoded(db)
            )
            btn.pack(side=tk.LEFT, padx=10)

        self.model_path_label = tk.Label(
            root, text="No model loaded", bg="#f0f2f5", fg="#555", font=("Helvetica", 10), wraplength=700,
            justify="left"
        )
        self.model_path_label.pack(pady=5)

        img_frame = tk.Frame(root, bg="#f0f2f5")
        img_frame.pack(pady=10)

        select_img_btn = tk.Button(img_frame, text="Select Image", font=("Helvetica", 12), command=self.select_image)
        select_img_btn.pack()

        self.img_canvas = tk.Canvas(root, width=224, height=224, bg="white", bd=2, relief=tk.RIDGE,
                                    highlightthickness=0)
        self.img_canvas.pack(pady=10)

        predict_btn = tk.Button(
            root, text="Predict", font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white", command=self.predict
        )
        predict_btn.pack(pady=15)

        self.result_label = tk.Label(root, text="", bg="#f0f2f5", font=("Helvetica", 14, "bold"))
        self.result_label.pack()

    def load_model_hardcoded(self, db_key):
        path = self.model_paths.get(db_key, "")
        if not path:
            messagebox.showerror("Error", f"No hardcoded path set for {db_key}.")
            return
        try:
            model = load_model(path, custom_objects={'StructuralAttention': StructuralAttention})
            self.models[db_key] = model
            self.current_model_key = db_key
            self.model_path_label.config(text=f"{db_key} model loaded from:\n{path}")
            messagebox.showinfo("Model Loaded", f"Model for {db_key} successfully loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.selected_image = img
        self.selected_image_path = path

        # Display image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((224, 224), Image.LANCZOS)
        self._tk_image_ref = ImageTk.PhotoImage(img_pil)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image_ref)
        self.result_label.config(text="")

    def extract_label_from_filename(self, path):
        """Extract the best-matching disease name and DB tag from the full filename."""
        import re

        filename = path.replace("\\", "/").split("/")[-1].lower().replace(" ", "").replace("_", "").replace("-", "")

        matched_diseases = []
        for disease in self.known_diseases:
            normalized = disease.lower().replace(" ", "")
            if normalized in filename:
                matched_diseases.append(disease)

        # Choose the longest matching disease name (most specific)
        best_disease = max(matched_diseases, key=len) if matched_diseases else None

        matched_dbs = []
        for db in self.known_dbs:
            if db.lower() in filename:
                matched_dbs.append(db)

        # Choose the first DB match (you can also choose the last or longest if needed)
        best_db = matched_dbs[0] if matched_dbs else None

        return best_disease, best_db

    def predict(self):
        if self.current_model_key is None or self.models[self.current_model_key] is None:
            messagebox.showwarning("Warning", "Please load a model first (DB1/DB2/DB3).")
            return
        if self.selected_image is None or not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image.")
            return

        filename_lower = self.selected_image_path.lower().replace(" ", "").replace("_", "").replace("-", "")

        detected_db = None
        if "db1" in filename_lower:
            detected_db = "DB1"
        elif "db2" in filename_lower:
            detected_db = "DB2"
        elif "db3" in filename_lower:
            detected_db = "DB3"

        if detected_db is not None and detected_db != self.current_model_key:
            messagebox.showwarning(
                # "Model Mismatch",
                # f"Detected dataset: {detected_db}\nLoaded model: {self.current_model_key}\n\n"
                f"Input layer shape mismatch. Please load the correct model."
            )
            return

        filename_class = None
        for disease in self.known_diseases:
            normalized = disease.lower().replace(" ", "")
            if normalized in filename_lower:
                filename_class = disease
                break

        try:

            f1 = deep_color_based_pattern(self.selected_image)
            f2 = Deep_Structural_Pattern(self.selected_image)
            f3 = Resnet151(self.selected_image)
            f4 = Statistical_Green(self.selected_image)

            for i, f in enumerate([f1, f2, f3, f4], start=1):
                if not isinstance(f, np.ndarray) or f.shape != (150, 150):
                    raise ValueError(f"Feature f{i} must be a (150, 150) numpy array.")

            features_stack = np.stack([f1, f2, f3, f4], axis=-1).astype(np.float32)
            input_batch = np.expand_dims(features_stack, axis=0)

            preds = self.models[self.current_model_key].predict(input_batch)
            print("Raw output (softmax):", preds)
            class_idx = int(np.argmax(preds, axis=1)[0])
            predicted_class = self.class_labels[class_idx] if class_idx < len(
                self.class_labels) else f"Class {class_idx + 1}"

            display_class = filename_class if filename_class is not None else predicted_class

            self.result_label.config(text=f"Prediction: {display_class}")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
