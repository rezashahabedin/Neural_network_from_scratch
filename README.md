reading hand written photo with neural networks without any enhanced libraries such as sklearn or tensorflow 
basically its a fully functional neural network built with raw python and numpy based on DR  Phil Tabor explanations which can be used freely for educational purposes .

number of hidden layer epoch, as well as learning rate can be tuned in code 

-line 104 	w1, w2, w3 = init_weights(784,100, 10)
784 is 28 by 28 which is the number of pixel each photo has in this sample 28 pixel vertically and 28 pixel horizontally 
100 is the number of hidden layers
10 is number of outputs, in this case we have 10 number so we need 10 output 

-line 105 	alpha= 0.001 
is learning rate which can be tuned

this code is dependent from any sikit-learn, tensorflow. or any other enhanced library in order to help understand the neural network

database is from yann.lecun.com
i couldn't log in to website any more since its asking user and password so i put dataset here as well
since its a byte data you need struct library to read data with python, depending on your cpu architecture you need to change symbol in struct.unpack command 
 struct.unpack('>IIII',img.read(16)) 
each 'I' stands for 4 byte, so we put 4 'I' for a 16 byte data
if your python fails to unpack data pls check struct website below and choose proper symbol for your system
https://docs.python.org/3/library/struct.html
