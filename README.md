# Facial-Emotion-Recognizer
## About 
The project involves detection of Human emotions from real time video streams, either from videos or from web cam videocapture based on the principle of Convolutional Neural Network(CNN).<br> 
The model is trained with the help of `keras` sequential model, by using three sets of training layers an percentage accuracy obtained of around 70%. The model is trained in around 50 epochs(pass) each containing around 448 files to train.<br><br>
The model we train is stored with the help of a `JSON` file and a `h5` file(for storing the weights) which is then restored to be compared to the Captured face and used for the emotion detection.

## Tools and Libraries used
1. `cv2`
2. `Keras models`
3. `Keras layers`
4. `keras image generator`
5. `numpy`

## Demo
<img src="https://github.com/user-attachments/assets/363d58fa-d833-4982-82be-be501abc1ebe" width=700 height=400>
<img src="https://github.com/user-attachments/assets/eef6dc5f-615a-4b8d-931a-f1cfd884e0a9" width=700 height=400>
<img src="https://github.com/user-attachments/assets/b01e679e-2653-4cdf-8ad4-67d39b166883" width=700 height=400>
