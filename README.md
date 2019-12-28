

### Classifier 2: Neural Net
Here the model is trained from the trainsets and tested on the test data for correctly identifying the orientation of images with the rotation labels. The model is architectured to setting the parameters on forward activation and then forward propagation and then tuning these on the feedback adjusted through the feedback networks. The error is adjusted on the basis of a cross entropy error calculated for each set of propagations. 

->The learning rate is also reduced throughout the descent for the batches of data for smoother convergence. This implies the tuning is done through batch gradient descent. 

-> In the forward propagation, dropout has been added to avoid overfitting of data alongwith the activation functions of relu(step function) and sigmoid(a bit curved/smoothed step) which are randomly selected.

-> The softmax function for the outermost layer ranges the propagation output between 0 and 1. 

-> Used pickle to save the model

#### How the model works:
Training for the image sets of all the four label orientations(0,90,180,270) are stored from train-data.txt. The processing of data and tuning of parameters is done by the orient.py file. Output of the trained data is the trained data stored in nnet_model.txt through the pickle package.

#### Train model has to be passed the following arguments:
./orient.py train train-data.txt nnet_model.txt nnet

#### Model is tested on arguments:
./orient.py test test-data.txt nnet_model.txt nnet

####  Observations:
-> The changing of learning rate in the batches improves the accuracy as it resulted in a smoother convergence.

-> Addition of droupouts regularized the output leading to reduction of the problem of overfitting of data.

-> The outputs with Relu and softmax activations work the best.


### Classifier 3:K nearest neighbors(KNN):

K nearest neighbors or KNN Algorithm is a simple classification algorithm which uses the entire dataset in its training phase. It uses the k value and distance metric (Euclidean distance) to measure the distance of new points to nearest neighbor. Whenever a prediction is required for an unseen data instance, it searches through the entire training dataset for k-most similar instances and the data with the most similar instance is finally returned as the prediction. The smaller the k value the greater the noise with the data; however, we can smooth this out by increasing the value of k.
The algorithm uses the neighbor points information to predict the target class. Following are the steps involved in KNN algorithm
1.	Getting Data
2.	Train & Test Data split
3.	Euclidean distance calculation
4.	Knn prediction function
5.	Accuracy calculation

Algorithm :   

In this assignment, we are reading the training file and storing the file in a model file along with the orientation and the RGB vector list. We are using this model file for Euclidean distance calculation and test data orientation prediction. After that 
1.	Pick a value for K.
2.	Take the K nearest neighbors of the new data point according to their Euclidean distance and assign an orientation for each of the image.
3.	Among these neighbors, count the number of data points in each category and assign the new data point to the category where you counted the most neighbors and their corresponding orientation.
4.	As the value of K goes higher, we stop at a K-value after which we don’t see much of improvement.
