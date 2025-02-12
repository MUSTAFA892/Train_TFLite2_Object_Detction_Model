
---

# Train_TFLite2_Object_Detection_Model

This repository contains a Jupyter notebook for custom object detection model training using TensorFlow Lite (TFLite). The process involves preparing data, configuring a TensorFlow object detection pipeline, training a custom model, and converting it to TensorFlow Lite format for efficient deployment on edge devices.

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Converting to TensorFlow Lite](#converting-to-tensorflow-lite)
5. [Inference with TFLite](#inference-with-tflite)

---

## Installation

Before starting the training process, ensure the necessary libraries and dependencies are installed:

```bash
pip install tensorflow tensorflow-hub tensorflow-object-detection-api tflite
```

---

## Data Preparation

1. **Dataset Collection**: Collect your images and label them using a tool like LabelImg to create `.xml` annotation files in PASCAL VOC format or use `.csv` annotations.
   
2. **Convert Annotations**: Convert the annotations to TensorFlow’s `TFRecord` format for compatibility with the training pipeline. Use the provided `generate_tfrecord.py` script.

```bash
python generate_tfrecord.py --images_path=images/ --annotations_path=annotations/ --output_path=output/
```

3. **Split Data**: Split your dataset into training and testing sets. Make sure to specify the paths in the pipeline config.

---

## Training the Model

1. **Pipeline Configuration**: Adjust the TensorFlow object detection pipeline configuration file to suit your custom dataset. Important parameters to modify:
   - `num_classes`: Set the number of object classes.
   - `train_input_path`: Path to the training TFRecord file.
   - `eval_input_path`: Path to the testing TFRecord file.

2. **Run the Training**: Once the configuration is set, start the training process using the following command:

```bash
python model_main_tf2.py --pipeline_config_path=path_to_pipeline_config --model_dir=training/ --alsologtostderr
```

3. **Monitor the Training**: The model will be saved periodically in the `model_dir`. You can monitor the training with TensorBoard:

```bash
tensorboard --logdir=training/
```

---

## Converting to TensorFlow Lite

After training the model, convert it to the TensorFlow Lite format for deployment on edge devices:

1. **Export SavedModel**: Once training is complete, export the model as a `SavedModel`.

```bash
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path=path_to_pipeline_config --trained_checkpoint_dir=training/ --output_directory=saved_model/
```

2. **Convert to TensorFlow Lite**: Convert the exported SavedModel to TensorFlow Lite format using the following code:

```python
import tensorflow as tf

# Load the SavedModel
saved_model_dir = 'saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Inference with TFLite

Once the model is converted to TensorFlow Lite format, use it for inference on edge devices. Here's a simple example of how to use the `.tflite` model for inference:

```python
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Input tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])
```

---

