import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained generator model
generator = tf.keras.models.load_model("generator_model.h5", compile=False)

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

def generate_images(selected_labels, n_per_label=5):
    label_indices = [class_names.index(lbl) for lbl in selected_labels]
    noise = tf.random.normal([n_per_label * len(label_indices), 100])
    
    label_vectors = []
    for label in label_indices:
        one_hot = tf.keras.utils.to_categorical([label] * n_per_label, 10)
        label_vectors.append(one_hot)
    label_vectors = np.vstack(label_vectors)
    
    generated_images = generator.predict([noise, label_vectors], verbose=0)
    generated_images = (generated_images + 1) / 2.0  # Scale to [0, 1]

    fig, axes = plt.subplots(len(label_indices), n_per_label, figsize=(n_per_label, len(label_indices)))
    if len(label_indices) == 1:
        axes = np.expand_dims(axes, 0)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(generated_images[i * n_per_label + j, :, :, 0], cmap='gray')
            ax.axis('off')
        row[0].set_ylabel(selected_labels[i], rotation=0, labelpad=40)
    plt.tight_layout()
    
    return fig

# Gradio Interface
label_choices = gr.CheckboxGroup(
    choices=class_names,
    label="Select Fashion Items",
    info="Choose one or more categories to generate samples."
)

interface = gr.Interface(
    fn=generate_images,
    inputs=label_choices,
    outputs=gr.Plot(label="Generated Images"),
    title="Conditional GAN: Fashion MNIST Generator",
    description="Select clothing categories and generate images using a pre-trained Conditional GAN (CGAN)."
)

interface.launch()
