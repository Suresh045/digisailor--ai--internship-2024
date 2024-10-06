import tensorflow as tf
import numpy as np
import cv2

# Load the MobileNetV2 model pre-trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to decode predictions
def decode_predictions(predictions):
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the model
        processed_frame = preprocess_image(frame)

        # Predict the object
        predictions = model.predict(processed_frame)
        decoded_predictions = decode_predictions(predictions)

        # Get the predicted object name
        object_name = decoded_predictions[0][1]

        # Display the prediction on the frame
        cv2.putText(frame, f'Object: {object_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Object Identification', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
