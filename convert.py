import tensorflow as tf
import os
import shutil

print("="*50)
print("🔄 Converting Model")
print("="*50)

# Load trained model
print("\n📂 Loading trained model...")
model = tf.keras.models.load_model("esrelive_model.h5")
print("✅ Model loaded")

# Output directory
output_dir = "web_model"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Export model in TensorFlow format
print("\n🔄 Exporting model...")
model.export(output_dir)

# Copy class labels file
if os.path.exists("class_labels.json"):
    shutil.copy("class_labels.json", os.path.join(output_dir, "class_labels.json"))
    print("✅ Copied class labels")

print("\n✅ Conversion complete!")
print(f"📁 Model saved to: {output_dir}/")