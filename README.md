# EmoNet

##

## 1. Introduction

Since the outbreak of COVID-19, we have been living with the pandemic for over two years. As universities moved to online learning, video meetings replaced in-person learning and became the norm. 

Even though most of the instructors would invite participants to turn on their webcams, we have found out that in actual meetings, most students are reluctant to show their faces. As such, online learning loses a very important part of teaching feedback which is facial expression. 

To allow lecturers to have a direct view of how well students are learning, and for students who want to participate but do not want to show their faces, our team is proposing EmoNet, an emoji generator based on facial expressions. Our model will generate an emoji based on the participant’s real-time facial expression, and display the emoji instead of cold, emotionless text. 





<img width="239" alt="image" src="https://user-images.githubusercontent.com/70104294/192652919-c6835035-2435-4e02-b1cc-02115d7b97a4.png">


Figure 1.1: Situations in online meetings before and after implementing our model [1]_

As millions of students use zoom to take lessons worldwise, manually labeling emotions is simply unfeasible. By using machine learning and using datasets to train our model, such tasks become easier to implement. During a zoom lesson, with the students’ consent, our model captures the student’s face at a certain frequency (i.e. every minute), uploads the captured face to our model, then outputs the emoji accordingly. Hence, the machine learning approach also ensures a timely update without exhausting resources.


## 


## 2. Illustration / Figure



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)





Figure 2.1: High-level illustration of model_


## 3. Background & Related Work

To achieve facial expression recognition, three processes have to take place: face acquisition, feature extraction, and expression recognition [2]. In the dataset that we acquired, the faces are already detected and cropped to appropriate dimensions, therefore our team will perform the feature extraction and expression recognition parts. However, during face acquisition, there are also two important steps: facial recognition and head pose detection [2], both of them which have been implemented with maturity. 

In_ Regularized Transfer Boosting for Face Detection Across Spectrum_, Zhang et al. proposed a multispectral face detecting algorithm using regularized transfer booting with manifold regularization. The algorithm was tested at 350 nm and 850 nm (both are non-visible light domains) and achieved significant improvement [3]. 

In _Head Pose Estimation in Computer Vision: A Survey_, [Erik Murphy-Chutorian](https://ieeexplore.ieee.org/author/38275910400), and [Mohan Manubhai Trivedi](https://ieeexplore.ieee.org/author/37271845000) discussed the difficulty of detecting head pose using computer vision systems and  composed an organized comparison between different approaches which spans over 90 papers that have been published on this topic [4]. 


## 


## 4. Data Processing


### 4.1 Dataset

The basic dataset we choose is FER2013, obtained from the Kaggle platform [5]. It contains 35886 48*48 pixel facial images with emotion labels. These images are all cropped well to have faces in the center. Samples are originally divided into seven classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. However, the number distribution between different classes is highly unbalanced, and the ratio between training, validation, and testing datasets are not perfect either.



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")


_Figure 4.1: Distribution of image numbers between classes in the original dataset_


### 4.2 Data Processing

As the disgust dataset is significantly lacking images, our team decided to formalize the dataset ratio to be 7.5: 1.5: 1.5. Using 3000 as our reference point, we calculated the number of images needed in validation and test datasets to be 642. 

To achieve such a number, our team decided to augment where the number of images does not reach our goal, and delete where the number exceeds our goal. To augment the dataset, our team chose a random direction (clockwise or counterclockwise), then randomly rotated by 5 - 30 degrees, incrementing by 5 degrees. After rotating the image, we zero-padded the external area, then center-cropped the image. 

_Figure 4.2: Sample of dataset augmentation_

After testing our early models, we have found that all of our models confuse “angry” with “disgust”, so we decided to combine angry with disgust as a single dataset. To combine these two, we replaced the last half of the disgust dataset with the first half of the angry dataset, while keeping the overall the same. Thus, a balanced dataset is achieved. 



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


_Figure 4.3: Balanced final dataset_

 


## 


## 5. Architecture

The primary model consists of two parts, the pre-trained ResNet transfer learning layers and classification layers (Figure 5.1).



<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


_Figure 5.1: Structure of the primary model_

The pre-trained ResNet model we use is ResNet50. All the inputs will be pre-processed by the ResNet50 and generate features of tensor type.  

The output is connected to the second part of the model, a classifier constructed by 3 fully connected layers. These layers will provide the final prediction among six facial expression categories.  

The neuron sizes of these three fully connected layers are 600, 50, and 6, respectively. 

Random dropouts are applied to the input and hidden layers, where the dropout rate is 0.2 for both layers. This dropout is used to address the potential overfitting. 

ReLU activation function is applied to the input layer and hidden layer in order to introduce non-linearities. 


## 


## 6. Baseline Model

The baseline model is a multi-class classifier consisting of CNN and NN layers (Figure 6.1). Our team believes this is a reasonable choice of baseline model since CNN-based classifiers have been proved to be effective in classification and are widely used.



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


_Figure 6.1: Structure of the baseline model_

The input of the baseline model is the FER2013 dataset that has been pre-processed. Our team implemented a CNN structure to extract facial expression-related features. There are two convolutional layers in the structure: 



* The first layer uses 10 of 5 * 5 size kernels to extract features from the input. 
* The second layer uses 20 of 5 * 5 size kernels to condense information. 
* After both layers, a max-pooling layer of 2 * 2 size kernels and 2 size stride is implemented to filter redundant features

The output of the second max-pooling layer is directly connected to the classifier consisting of two fully connected layers. The neuron sizes of these two fully connected layers are 32, and 6, respectively. 

Tuning the baseline model, we get the hyperparameters of the best model: 



* batch size = 64
* learning rate = 2e-5
* number of epochs = 16

With these hyperparameters, the test accuracy of the baseline model is 36.1%. Since the reported best accuracy of the FER2013 dataset is 76.82% [6], our team believes we have achieved a reasonable baseline accuracy on our dataset.  


## 


## 7. Result


### 7.1 Quantitative Result

The hyperparameters of our best primary model are



* batch size = 32
* learning rate = 5e-4
* number of epochs = 12

We chose F1 score and accuracy as metrics for our primary model. F1 score refers to the harmonic mean of the precision and recall of the model where precision is the fraction of the true positive and retrieved elements, and recall is the fraction of the true positive and relevant elements. Accuracy shows the proportion of the correctly classified expressions. 

The visualized metrics of both training and validation sets are shown in Figure 7.1 - 7.2. The model has a final training accuracy of 92.34% and validation accuracy of 58.67%. The final training F1 score is 0.75 and the validation F1 score is 0.59. 



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.jpg "image_tooltip")


<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.jpg "image_tooltip")


_Figure 7.1-7.2: Accuracy and F1 Score curves_


### 


### 7.2 Qualitative Result

To see how the model performs more intuitively, here are some sample outputs with both real and predicted labels. In this batch of randomly selected images, some are easy to determine while others are hard to tell by human experts. Hence, these eight images are relatively representative.



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")


_Figure 7.3: Sample outputs with real and predicted labels_

In order to further analyze the accuracies between different types of input, we applied a confusion matrix generated from a part of our validation dataset. It comprehensively demonstrates the model’s ability to classify different emotions. We can draw several conclusions from calculation and observation:



* The model achieves the highest prediction accuracy on class “happy” (83.6%) and also performs well with another positive emotion “surprise” (72.8%). On the other hand, it is not very good at recognizing negative emotions such as “fear” and “sad”, with the accuracy of 44.1% and 50.9% respectively.
* In terms of possible reasons for incorrect prediction, we noticed that the model is likely to confuse emotions with similar inclinations (i.e. Given “fear” but predicts “disgust”), and that it tends to generalize emotions without exaggerated features to neutral (i.e. Given “sad” but predicts “neutral”).



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.jpg "image_tooltip")


_Figure 7.4: Confusion matrix_

Based on the mechanism behind the model, we can make an inference that if an input image has more obvious and significant features, the accuracy of prediction will be improved.


## 8. Evaluate Model on New Data

The testing dataset was prepared as described in section 4.2. It was never used for the purpose of training or parameter-tuning. The final model achieved an overall accuracy of 66.45% on this testing dataset. 

The goal of this project is to represent participants in the form of emojis, hence besides the testing dataset, we gathered extra photos and tried to simulate real lecture scenarios. However, since some of the emotions are hard to represent and rarely seen in lectures or meetings, only parts of the extra test photos were taken by the team members. For other rarely seen emotion classes, we looked online for sample photos from authorized websites. The following four figures are some of the correct predictions. As shown in the results, the model correctly classified “happy”, “disgust”, “neutral” and “fear” emotions from the input images. To have a direct view of the model’s mechanism, we have also placed the classification score of each class for a given image in the plot. 

Overall, EmoNet achieved a 62% accuracy on the extra dataset, which matches our expectation of testing accuracy. 

_Figure 8.1 - 8.4: some correct predictions made by EmoNet_


## 9. Discussion

Based on the previous confusion matrix, here we replaced all numbers with their percentages calculated both horizontally (Table 9.1) and vertically (Table 9.2), and interpreted them in a more specific way.


<table>
  <tr>
   <td rowspan="2" colspan="2" >
   </td>
   <td colspan="7" >Predicted Labels
   </td>
  </tr>
  <tr>
   <td>Disgust
   </td>
   <td>Fear
   </td>
   <td>Happy
   </td>
   <td>Neutral
   </td>
   <td>Sad
   </td>
   <td>Surprise
   </td>
   <td>Total
   </td>
  </tr>
  <tr>
   <td rowspan="6" >Real
<p>
 Labels
   </td>
   <td>Disgust
   </td>
   <td>68
   </td>
   <td>8
   </td>
   <td>2
   </td>
   <td>6
   </td>
   <td>12
   </td>
   <td>4
   </td>
   <td>100
   </td>
  </tr>
  <tr>
   <td>Fear
   </td>
   <td>15
   </td>
   <td>44
   </td>
   <td>5
   </td>
   <td>17
   </td>
   <td>11
   </td>
   <td>8
   </td>
   <td>100
   </td>
  </tr>
  <tr>
   <td>Happy
   </td>
   <td>1
   </td>
   <td>3
   </td>
   <td>84
   </td>
   <td>5
   </td>
   <td>3
   </td>
   <td>4
   </td>
   <td>100
   </td>
  </tr>
  <tr>
   <td>Neutral
   </td>
   <td>3
   </td>
   <td>6
   </td>
   <td>7
   </td>
   <td>69
   </td>
   <td>12
   </td>
   <td>3
   </td>
   <td>100
   </td>
  </tr>
  <tr>
   <td>Sad
   </td>
   <td>11
   </td>
   <td>11
   </td>
   <td>4
   </td>
   <td>22
   </td>
   <td>51
   </td>
   <td>1
   </td>
   <td>100
   </td>
  </tr>
  <tr>
   <td>Surprise
   </td>
   <td>9
   </td>
   <td>6
   </td>
   <td>3
   </td>
   <td>5
   </td>
   <td>4
   </td>
   <td>73
   </td>
   <td>100
   </td>
  </tr>
</table>


_Table 9.1: Horizontal interpretation of the validation subset results (All numbers are in percentage)_


<table>
  <tr>
   <td rowspan="2" colspan="2" >
   </td>
   <td colspan="6" >Predicted Labels
   </td>
  </tr>
  <tr>
   <td>Disgust
   </td>
   <td>Fear
   </td>
   <td>Happy
   </td>
   <td>Neutral
   </td>
   <td>Sad
   </td>
   <td>Surprise
   </td>
  </tr>
  <tr>
   <td rowspan="7" >Real
<p>
 Labels
   </td>
   <td>Disgust
   </td>
   <td>60
   </td>
   <td>9
   </td>
   <td>1
   </td>
   <td>5
   </td>
   <td>11
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td>Fear
   </td>
   <td>14
   </td>
   <td>54
   </td>
   <td>4
   </td>
   <td>13
   </td>
   <td>12
   </td>
   <td>7
   </td>
  </tr>
  <tr>
   <td>Happy
   </td>
   <td>1
   </td>
   <td>5
   </td>
   <td>82
   </td>
   <td>5
   </td>
   <td>4
   </td>
   <td>4
   </td>
  </tr>
  <tr>
   <td>Neutral
   </td>
   <td>3
   </td>
   <td>7
   </td>
   <td>6
   </td>
   <td>55
   </td>
   <td>12
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>Sad
   </td>
   <td>12
   </td>
   <td>16
   </td>
   <td>4
   </td>
   <td>18
   </td>
   <td>56
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>Surprise
   </td>
   <td>10
   </td>
   <td>9
   </td>
   <td>3
   </td>
   <td>4
   </td>
   <td>5
   </td>
   <td>81
   </td>
  </tr>
  <tr>
   <td>Total
   </td>
   <td>100
   </td>
   <td>100
   </td>
   <td>100
   </td>
   <td>100
   </td>
   <td>100
   </td>
   <td>100
   </td>
  </tr>
</table>


_Table 9.2: Vertical interpretation of the validation subset results (All numbers are in percentage)_

Table 9.1 shows the distribution of predicted labels regarding the real labels and Table 9.2 shows the distribution of real labels regarding predicted labels. The values on the diagonal axes of Table 9.1 and 9.2 are marked in darker colors, as they represent the recalls and precisions of the results respectively.

According to Table 9.1, the model has error rates of 32%, 56%, 16%, 31%, 49%, and 27% corresponding to classes “disgust”, “fear”, “happy”, “neutral”, “sad” and “surprise”. Detailed percentages of wrong predictions are presented in the table.

It can be concluded that the model seldomly falsely predicts positive emotions “happy” and “surprise”, unlike when the model predicts negative emotions such as “fear” and “sad”. This implies that the model is not much likely to recognize a face with positive emotion to be negative, which means if a user of EmoNet is listening to a video conference with interest and no confusion, their emotion will not be displayed as negative. This feature prevents the instructor from receiving falsely displayed negative emotions, therefore preserving the pace of lecturing. 

Also as mentioned in section 7.2, we noticed that when the real label is “fear”, 26% are predicted as “disgust” and “sad”, and 17% are predicted as “neutral”; when the real label is “sad”, 22% are predicted as “disgust” and “fear”, and 22% are predicted as “neutral”. It means that although the model has relatively high error rates when recognizing negative emotions, it is mainly due to confusion with other negative and neutral expressions, rather than positive emotions. This feature reduces the impact of EmoNet’s wrong outputs in practical video meetings.

From Table 9.2, it can be noted that the recalls of “fear” and “sad” are low, which implies an insensitivity to classify these emotions. Additionally, many facial expressions are debatable through the eyes of different people, such as the following images in Figures 9.1 - 9.2. Thus, we must admit that there are unavoidable limitations of such facial expression recognition tools in practice.

_Figure 9.1 - 9.2: Debatable facial expressions_


## 


## 10. Ethical Consideration

Our team strives to look for publicly available, authorized data. During training and testing, our model also needs to make sure that the pictures stay in the sample, and at no stage during the training will the data leak anywhere else. Our dataset also needs to include as many races and minority groups as possible to accommodate different facial features. 

Our model may contain several biases. Images from the FER2013 dataset are labeled manually, thus the labels may contain systematic measurement bias as it is hard for humans to recognize the emotion by only looking at a static photo. 

The training data and the testing dataset were randomly selected, therefore we could not guarantee that all representatives of different groups of the population were contained within the datasets. Therefore the model might have an evaluation bias due to the lack of equal representation of different groups. This may lead to a result that our model could have a bad performance over some specific under-represented groups during the testing phase. 

To fully test our models, we also need to collect images from as many races and groups of people as possible to see if there exist any biases, then re-train our model if any bias is encountered. 


## 11. Project Difficulty / Quality

This project intends to construct a communication channel between instructors and students who want to participate but do not want to show their faces. Our team spent some time augmenting the data in FER-2013 through rotation and re-organizing the processed data into six classes to reach a balanced dataset for achieving a fair model.

Our team started by implementing a baseline CNN model, then continuously improved the predicting accuracy through several different transfer-learning models, including AlexNet, VGG, GoogLeNet, and ResNet. During the training phase, we tuned different sets of hyperparameters and eventually chose ResNet as our final model with both the highest validation and testing accuracies.

To test the performance, we came up with two different methods. We first tested our model using a randomly selected dataset from FER-2013. Then to simulate real-life scenarios, we utilized the webcam and OpenCV to recognize our team members’ faces so that we can process them, feed them into our model, and check the model’s predictions.

The final model has an accuracy of 66.45%, and there is still potential to be improved upon to reach the best-recorded accuracy of 76.82% [6]. However, facial emotion recognition is a hard task itself. Research shows that human performance on this dataset is estimated to be 65.5 % [7], which our model outperforms. 

Therefore, due to the dataset manipulation, multiple model building, utilization of OpenCV to test real-life scenarios, and results exceeding human performance, we consider our project high in difficulty and completed with high quality. 




## 12. References

[1]    “Tech tips to feel connected to students, even when their cameras are off,” _Tech for Teaching_

_blog_. [Online]. Available: [https://t4t.blog.ryerson.ca/2020/09/15/tech-tips-to-feel-connected-to-students-even-when-their-cameras-are-off/](https://t4t.blog.ryerson.ca/2020/09/15/tech-tips-to-feel-connected-to-students-even-when-their-cameras-are-off/). [Accessed: 11-Apr-2022]. 

[2]    Q. Liu and J. Wang, “Facial expression recognition based on CNN -project jupyter.” [Online]. Available: [https://nbviewer.org/github/charlesliucn/miscellanea/blob/master/00-course-theses/17-Facial_Expression_Recognition.pdf](https://nbviewer.org/github/charlesliucn/miscellanea/blob/master/00-course-theses/17-Facial_Expression_Recognition.pdf). [Accessed: 09-Feb-2022].

[3]    Zhang, Zhiwei, et al. ”Regularized Transfer Boosting for Face Detection Across Spectrum.” IEEE Signal Processing Letters 19.3(2012):131-134.

[4]    E. Murphy-Chutorian and M. M. Trivedi, "Head Pose Estimation in Computer Vision: A Survey," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 4, pp. 607-626, April 2009, doi: 10.1109/TPAMI.2008.106.

[5]    M. Sambare, “FER-2013,” _Kaggle_, 19-Jul-2020. [Online]. Available: [https://www.kaggle.com/msambare/fer2013](https://www.kaggle.com/msambare/fer2013). [Accessed: 12-Apr-2022]. 

[6]    “Papers with code - fer2013 benchmark (facial expression recognition),” _The latest in Machine Learning_. [Online]. Available: [https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013). [Accessed: 12-Apr-2022]. 

[7]    Ian J. Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, Yingbo Zhou, Chetan Ramaiah, Fangxiang Feng, Ruifan Li, Xiaojie Wang, Dimitris Athanasakis, John Shawe-Taylor, Maxim Milakov, John Park, Radu Ionescu, Marius Popescu, Cristian Grozea, James Bergstra, Jingjing Xie, Lukasz Romaszko, Bing Xu, Zhang Chuang, Yoshua Bengio, “Challenges in representation learning: A report on three machine learning contests,” _Neural Networks_, vol. 64, p. 59 - 63, April, 2015. [Online Serial]. Available: [https://www.sciencedirect.com/science/article/pii/S0893608014002159#](https://www.sciencedirect.com/science/article/pii/S0893608014002159#). [Accessed: 12-Apr-2022]. 
