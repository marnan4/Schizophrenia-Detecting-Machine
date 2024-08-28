import numpy as np
import tensorflow as tf

def train_and_predict_brain_scans(new_brain_scans):
    # Generate sample data (replace this with your actual brain scan data)
    brain_scans_data = np.array([[0.8, 0.5],
                                 [0.3, 0.2],
                                 [0.6, 0.4],
                                 [0.2, 0.1]])
    labels = np.array([1, 0, 1, 0])

    # Normalize the data (optional but recommended for neural networks)
    brain_scans_data = (brain_scans_data - np.mean(brain_scans_data, axis=0)) / np.std(brain_scans_data, axis=0)

    # Split the data into training and testing sets
    num_samples = len(brain_scans_data)
    train_samples = int(0.8 * num_samples)  # 80% of the data for training
    x_train, x_test = brain_scans_data[:train_samples], brain_scans_data[train_samples:]
    y_train, y_test = labels[:train_samples], labels[train_samples:]

    # Build the neural network with more hidden layers and regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=1)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)

    # Normalize the new brain scan data using the same mean and standard deviation
    new_brain_scans = (new_brain_scans - np.mean(brain_scans_data, axis=0)) / np.std(brain_scans_data, axis=0)

    # Make predictions on new brain scans
    predictions = model.predict(new_brain_scans)

    # Interpret the results
    for i, prediction in enumerate(predictions):
        if prediction[0] >= 0.5:
            print(f"Scan {i + 1}: Schizophrenia likely")
        else:
            print(f"Scan {i + 1}: Schizophrenia unlikely")

# Example usage with new brain scan data

new_brain_scans = np.array([[0.9, 0.6]])
train_and_predict_brain_scans(new_brain_scans)
