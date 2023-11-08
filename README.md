# fake_face_detection
 Fake Face Detection System using Convolutional Neural Networks (CNN) with Python

 ## Subject

 We can basically define the subject of our project as "Fake Face Recognition with Deep Learning".
 
 In this project, deep learning techniques are used to distinguish fake and real faces. While facial recognition technologies are of great importance in social media platforms, games and many digital platforms, misuse of these technologies is also seen. Especially fake faces created with DeepFake technology can cause serious problems in many areas. This situation has revealed a need to automatically distinguish fake and real faces.

 ### Fake Face Detection

 Fake Face Detection is a technology used to determine whether a face is real or fake (synthetic or manipulated). Fake faces are often created using advanced machine learning techniques such as DeepFake technology. These technologies allow realistic superimposition of one person's face onto another person's face. Fake faces can be used in many areas, from social media accounts to video conferences and digital media content. Therefore, detecting fake faces and distinguishing them from real faces is of great importance.

 Fake Face Detection technology is often developed using techniques such as deep learning and Convolutional Neural Networks (CNN). These models learn the unique features of faces and then use those features to determine whether new images are real or fake.

 Fake Face Detection can help prevent persona theft, fraud, and the spread of misleading information. Additionally, fake face detection technology can also be used to filter misleading and manipulative content on digital media and social media platforms. This can protect users' security and privacy and help enforce community standards and laws.

 ## Purpose

 The aim of the project is to create a model that can analyze various facial images and determine whether these images are real or fake. For this purpose, a deep learning model was developed using Convolutional Neural Networks (CNN) technology.

 A large data set consisting of real and fake faces was used in training the model. This dataset enabled the model to effectively recognize both types of faces. During training and testing of the model, high accuracy rates were achieved.

 This project demonstrates the power of deep learning techniques and their potential to improve the security of various applications.

 ## Content

 **Data Import:** The project uses a dataset containing real and fake face images. Each of these images represents either a real face or a fake face.

 **Data Preparation:** All images in the data set are pre-processed, which may affect the learning process of the model. Images are resized to a size that the model can accept as input. Additionally, the color channels of the images are also edited.

 **Data Augmentation:** Data augmentation techniques are used to increase the generalization ability of the model. This includes operations such as rotating the images in the data set and flipping them horizontally and vertically.

 **Model Creation:** Within the scope of the project, a special CNN model was created based on MobileNetV2. Both models are used to learn features of images and classify fake and real faces. The models consist of various convolutional, pooling, and fully connected layers.

 In general, we execute the following in this model:

 We import a pre-trained version of MobileNetV2. The model is imported with the include_top=False parameter, which means the last Fully Connected (Dense) layer will not be in the model. Instead of this layer, we will add a specially designed output layer. Additionally, the weights are determined as pre-trained weights from the ImageNet dataset and the input shape of the model is determined as (224,224,3).

 **Editing the Model:** We add new layers to the MobileNetV2 model that add the ability to learn from the dataset. This customized model takes MobileNetV2 as input and includes the Flatten layer as output and ultimately a Dense layer using 'softmax' as the activation function. The Flatten layer reduces the outputs of convolutional layers to a single dimension. The Dense layer performs the classification process.

 **Freezing Layers:** We freeze the layers of the MobileNetV2 model. This means that the weights of these layers will not be updated during training. This step preserves the previously trained weights and ensures that only new layers added are trained.

 **Compiling the Model:** We compile the created model, here we use "sparse_categorical_crossentropy" as the loss function, "adam" as the optimizer and "accuracy" as the metric.

 This model uses convolutional and pooling layers to learn features of facial images, and uses a fully connected (dense) layer to distinguish fake and real faces through these features. During the training process, the model learns features and assigns a weight to each feature, these weights are used to make predictions. The performance of the model is evaluated by the accuracy rate in the training and validation data sets.

 **Model Training:** Models are trained on training data. In this process, the model learns the features of fake and real faces. During the training process, the model's performance is regularly tested on the validation dataset.

 **Performance Evaluation:** After the training of the model is completed, its performance on the training and validation data sets is examined through graphs.

 **Predictions:** Finally, the model is used to predict whether the faces in the data set are real or fake.

 ## Dataset

 The dataset used in this project is the "Real and Fake Face Detection" dataset provided by Kaggle. This dataset is a very valuable resource for machine learning models in the field of facial recognition and fraud detection.

 The dataset includes real and fake (manipulated) human faces. Fake faces were created using a variety of facial manipulation techniques. This makes our model robust against different types of fraud.

 The dataset contains two main folders: “training_real” and “training_fake”. Both folders contain .jpg files containing facial images. The "training_real" folder contains real faces, while the "training_fake" folder contains fake faces. Each facial image provides variation with different lighting conditions, angles and expressions.

 Each image in the dataset is resized to size (224, 224, 3) so that the model can identify faces more accurately. This sizing process ensures that the model renders all images at the same size.

 The main reason for choosing this dataset is that it allows the model to learn face recognition and fake face detection. This improves the overall performance of the model and results in more accurate facial fraud detection. This dataset directly supports the main goal in this project – namely, gaining the ability to distinguish between fake and real faces.

 ### Images from the Dataset

 #### Fake Faces
 
 <img width="631" alt="fakefaces" src="https://github.com/osmantunahanincirkus/fake_face_detection/assets/106384513/dbae4bd0-3120-406e-bf74-4b9b8e51aa82">

 #### Real Faces
 
 <img width="632" alt="realfaces" src="https://github.com/osmantunahanincirkus/fake_face_detection/assets/106384513/e55a45ac-a7bf-4aa6-a1d0-aa52a9a7c95f">

 ## Classification Performance

 While training our model, we observed the training and validation accuracies and losses after each epoch. This information was crucial to monitoring our model's performance and making adjustments as needed.

 In this project, we trained our model for 20 epochs and monitored the loss and accuracy values of the training and validation sets after each epoch. In the first epoch, our model was trained with 93.51% accuracy and achieved 100% accuracy on the validation set. This showed that our model was very effective at distinguishing fake and real faces.

 In subsequent epochs, the accuracy values of our model consistently reached 100%. This shows that our model correctly classified all training and validation images. Additionally, loss values continued to decrease with each epoch. This shows that our model's predictions are becoming increasingly reliable.

 In the last epoch, the training accuracy of our model was 100% and the validation accuracy was 100%. These results show that our model perfectly classifies images in both the training and validation set.

 As a result, the model we created and trained in this project performed with high accuracy in distinguishing fake and real faces. This shows that our model is well suited for this type of classification task.

 ## Graphs

 ### Accuracy Graph

 <img width="505" alt="accgraph" src="https://github.com/osmantunahanincirkus/fake_face_detection/assets/106384513/65acb3ca-13aa-4d57-a21e-c8b2a2a662c2">

 ### Loss Graph

 <img width="505" alt="lossgraph" src="https://github.com/osmantunahanincirkus/fake_face_detection/assets/106384513/3ca5829f-bbf8-4b84-ac29-df0f7790ae69">

 ## Result

 Throughout this project, we built and trained a deep learning model capable of distinguishing between fake and real faces. Our model was based on VGG16, an advanced Convolutional Neural Network (CNN) architecture, and learned complex features by creating a multilayer neural network.

 During the training process, our model achieved high accuracy rates on both training and validation sets, indicating that the overall performance of the model and its ability to distinguish real and fake faces are extremely high.

 Our complexity matrix analysis also confirmed the success of our model in accurately predicting both classes. This shows that the model performs flawlessly when distinguishing between fake and real faces.

 As a result, we have successfully solved an important problem arising from the combination of fake face detection, deep learning and image processing technologies.

 ## Contribution

Thank you for your contributions.

- [@IsmailCanMutlu](https://github.com/IsmailCanMutlu)

## License

[MIT License](LICENSE)
