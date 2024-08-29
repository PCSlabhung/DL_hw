from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.fc1 = FullyConnected(28*28, 392) ## Just an example.You can alter sample code anywhere. 
        self.act1 = ACTIVITY1()
        self.fc2 = FullyConnected(392, 196)
        self.act2 = ACTIVITY1()
        self.fc3 = FullyConnected(196,98)
        self.act3 = ACTIVITY1()
        self.fc4 = FullyConnected(98,49)
        self.act4 = ACTIVITY1()
        self.fc5 = FullyConnected(49,10)
        self.loss = SoftmaxWithloss()



    def forward(self, input, target):
        h1 = self.fc1.forward(input)
        ## by yourself .Finish your own NN framework
        A1 = self.act1.forward(h1)
        h2 = self.fc2.forward(A1)
        A2 = self.act2.forward(h2)
        h3 = self.fc3.forward(A2)
        A3 = self.act3.forward(h3)
        h4 = self.fc4.forward(A3)
        A4 = self.act4.forward(h4)
        h5 = self.fc5.forward(A4)
        pred, loss = self.loss.forward(h5,target)
        return pred, loss

    def backward(self):
        ## by yourself .Finish your own NN framework
        A5_grad = self.loss.backward()
        h5_grad = self.fc5.backward(A5_grad)
        A4_grad = self.act4.backward(h5_grad)
        h4_grad = self.fc4.backward(A4_grad)
        A3_grad = self.act3.backward(h4_grad)
        h3_grad = self.fc3.backward(A3_grad)
        A2_grad = self.act2.backward(h3_grad)
        h2_grad = self.fc2.backward(A2_grad)
        A1_grad = self.act1.backward(h2_grad)
        h1_grad = self.fc1.backward(A1_grad)

    def update(self, lr):
        ## by yourself .Finish your own NN framework
        
        self.fc1.weight -= self.fc1.weight_grad * lr
        self.fc1.bias -= self.fc1.bias_grad * lr
        self.fc2.weight -= self.fc2.weight_grad * lr
        self.fc2.bias -= self.fc2.bias_grad * lr
        self.fc3.weight -= self.fc3.weight_grad * lr
        self.fc3.bias -= self.fc3.bias_grad * lr
        self.fc4.weight -= self.fc4.weight_grad * lr
        self.fc4.bias -= self.fc4.bias_grad * lr
        self.fc5.weight -= self.fc5.weight_grad * lr
        self.fc5.bias -= self.fc5.bias_grad * lr
        
        
