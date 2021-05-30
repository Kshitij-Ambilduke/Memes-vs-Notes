# Memes-vs-Notes

This repository contains the implementation of a binary classifier which was implemented on a self made dataset of Memes and Notes where the model was made to make a prediction whether a given image is a Meme or Notes taken from that boring class you took. <br />
This binary classification was done using 2 methods namely a DNN (Dense neural network) and a CNN (Convolutional neural Network). The Dense Neural network was implemented twice, once from scratch using scientific libraries in python (like NumPy and PIL) and for the second time using the deeplearning framework **PyTorch**.  <br />
The Convolutional Neural Network was implemented completely in PyTorch along with some other libraries for preprocessing and cleaning the dataset.  <br />

### Dense Neural Networks : 
A Dense Neural Network is a family of neural networks in which each neuron in each layer is connected to each neuron in the next layer. The connections within these neurons are controlled by the weights and biases of the network accounting for the parameters of the network. In a general fashion, the DNN in case of an image input, takes each pixel as input and by tweaking and twisting its parameters, it learns a function that maps a particular input to a particular output.  <br />
Following is the architecture followed for the task at hand :  <br />
```
DenseNet(
  (dense1): Linear(in_features=30000, out_features=10000, bias=True)
  (dense2): Linear(in_features=10000, out_features=5000, bias=True)
  (dense3): Linear(in_features=5000, out_features=1000, bias=True)
  (dense4): Linear(in_features=1000, out_features=500, bias=True)
  (dense5): Linear(in_features=500, out_features=100, bias=True)
  (dense6): Linear(in_features=100, out_features=50, bias=True)
  (dense7): Linear(in_features=50, out_features=2, bias=True)
)
```
**Following are the results achieved in terms of accuracy and loss :**  <br />
 <br />
![](https://github.com/Kshitij-Ambilduke/Memes-vs-Notes/blob/master/DNN.PNG)

### Convolutional Neural Networks : 
A convolutional neural network (CNN) is a specialized family of neural networks which specializes in the vision domain. The main striking difference between the Dense neural network and a CNN is that CNN uses weight sharing and kernals which learns the important characteristics in the image thereby learning the semantic relations in the images. Owing to all this as expected, better results are seen using this network. A more detailed explaination of this network is provided in the notebook. <br />
Following is the architecture followed for the task at hand :  <br />
```
ConvNet(
  (l1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l3): Sequential(
    (0): Conv2d(32, 48, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (l4): Sequential(
    (0): Linear(in_features=3888, out_features=500, bias=True)
    (1): Linear(in_features=500, out_features=2, bias=True)
  )
)
```
**Following are the results achieved in terms of accuracy and loss :**  <br />
 <br />
![](https://github.com/Kshitij-Ambilduke/Memes-vs-Notes/blob/master/cc.PNG)

