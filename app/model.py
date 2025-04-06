import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os

def create_model():
    """
    Create CNN model architecture
    
    Returns:
    - Compiled Keras model
    """
    model = models.Sequential([
        Input(shape=(28, 28, 1)),  # Proper input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model():
    """
    Train and save the CNN model
    """
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess data
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    
    # Create and train model
    model = create_model()
    model.fit(
        train_images, 
        train_labels, 
        epochs=5, 
        validation_data=(test_images, test_labels)
    )
    
    # Save model
    os.makedirs('app', exist_ok=True)
    model.save('app/mnist_cnn.h5')
    return model

if __name__ == '__main__':
    train_and_save_model()