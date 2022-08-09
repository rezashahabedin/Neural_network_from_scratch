import struct 
import numpy as np
import matplotlib.pyplot as plt
import os
def load_data():
	with open('data/train-labels-idx1-ubyte','rb') as labels:
		magic, n= struct.unpack('>II',labels.read(8))
		train_labels= np.fromfile(labels, dtype=np.uint8)

	with open('data/train-images-idx3-ubyte','rb') as img:
		magic, num, nrows, ncols= struct.unpack('>IIII',img.read(16))
		train_images= np.fromfile(img, dtype = np.uint8).reshape(num,784)

	with open('data/t10k-labels-idx1-ubyte','rb') as labels:
		magic, num= struct.unpack('>II',labels.read(8))
		test_labels= np.fromfile(labels, dtype = np.uint8)

	with open('data/t10k-images-idx3-ubyte','rb') as img:
		magic, num, nrows, ncols= struct.unpack('>IIII',img.read(16))
		test_images= np.fromfile(img, dtype = np.uint8).reshape(num,784)
	return train_images, train_labels, test_images, test_labels


def enc_one_hot (y, num_labels=10):
	one_hot = np.zeros((num_labels, y.shape[0]))
	for i, val in enumerate(y):
		one_hot[val,i] = 1
	return one_hot

def sigmoid(z):
	return (1 / (1+ np.exp(-z)))


def sigmoid_gradient(z):
	s= sigmoid(z)
	return s *(1-s)


def calculate_cost(y_enc, outpt):
	t1= -y_enc * np.log(outpt)	
	t2= (1 - y_enc) * np.log(1 - outpt)
	cost= np.sum(t1 - t2)
	return cost

def add_bias_unit(x, where):
	if where=='column':
		x_new= np.ones((x.shape[0],x.shape[1]+1))

		x_new[:,1:] =x
	elif where=='row':
		x_new= np.ones((x.shape[0]+1,x.shape[1]))
		x_new[1:,:] =x
	return x_new

def init_weights(n_features, n_hidden, n_output):
	w1= np.random.uniform(-1.0, 1.0, size= n_hidden * (n_features+ 1))
	w1= w1.reshape(n_hidden, n_features+1)
	w2= np.random.uniform(-1.0, 1.0, size= n_hidden * (n_hidden+ 1))
	w2= w2.reshape(n_hidden, n_hidden+1)
	w3= np.random.uniform(-1.0, 1.0, size= n_output * (n_hidden+ 1))
	w3= w3.reshape(n_output, n_hidden+1)
	return w1,w2,w3

def feed_forward(x, w1, w2, w3):
	a1= add_bias_unit(x, where='column')
	z2= w1.dot(a1.T)

	a2= sigmoid(z2)
	a2= add_bias_unit(a2,where='row')
	z3= w2.dot(a2)

	a3= sigmoid(z3)
	a3= add_bias_unit(a3, where='row')
	z4= w3.dot(a3)

	a4= sigmoid(z4)
	return a1, z2, a2, z3, a3, z4, a4

def predict(x, w1, w2, w3 ):
	a1, z2, a2, z3, a3, z4, a4= feed_forward(x, w1, w2, w3)
	y_pred= np.argmax(a4, axis=0)
	return y_pred

def calc_gradient(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
	delta4= a4-y_enc
	z3= add_bias_unit(z3, where='row')
	delta3= w3.T.dot(delta4)*sigmoid_gradient(z3)
	delta3= delta3[1:,:]
	z2=add_bias_unit(z2, where='row')
	delta2= w2.T.dot(delta3)*sigmoid_gradient(z2)
	delta2= delta2[1:,:]

	grad1= delta2.dot(a1)
	grad2= delta3.dot(a2.T)
	grad3= delta4.dot(a3.T)
	return grad1, grad2, grad3

def run_model(x, y, x_t, y_t):
	x_copy, y_copy= x.copy(), y.copy()
	y_enc= enc_one_hot(y)
	epochs= 200
	batch= 25

	w1, w2, w3 = init_weights(784,100, 10)
	alpha= 0.001
	eta= 0.001
	dec= 0.00001
	delta_w1_prev= np.zeros(w1.shape)
	delta_w2_prev= np.zeros(w2.shape)
	delta_w3_prev= np.zeros(w3.shape)
	total_cost=[]
	pred_acc= np.zeros(epochs)

	for i in range(epochs):
		
		shuffle= np.random.permutation(y_copy.shape[0])
		x_copy, y_enc= x_copy[shuffle], y_enc[:,shuffle]
		eta /= (1+ dec*i)

		mini= np.array_split(range(y_copy.shape[0]),batch)

		for step in mini:
			a1, z2, a2, z3, a3, z4, a4 =feed_forward(x_copy[step], w1, w2, w3)
			cost= calculate_cost(y_enc[:,step],a4)

			total_cost.append(cost)

			grad1, grad2, grad3= calc_gradient(a1, a2, a3, a4, z2, z3, z4, y_enc[:,step], w1, w2, w3)
			delta_w1, delta_w2, delta_w3 = eta*grad1, eta*grad2, eta*grad3

			w1-= delta_w1+alpha*delta_w1_prev
			w2-= delta_w2+alpha*delta_w2_prev
			w3-= delta_w3+alpha*delta_w3_prev

			delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3
		
		y_pred=predict(x_t,w1,w2,w3)
		pred_acc[i] =100*(np.sum(y_t == y_pred, axis=0) / x_t.shape[0])
		print( 'epochs number: ',i, ' ,accuracy is: ',pred_acc[i])
	return total_cost, pred_acc,y_pred

traint_x, train_y, test_x, test_y = load_data()
cost, acc,y_pred= run_model(traint_x, train_y, test_x, test_y)
x_a=[i for i in range(acc.shape[0])]
x_c=[i for i in range(len(cost))]

fig, (ax1, ax2) = plt.subplots(1, 2)
print('best accuracy : ',acc[-1])
ax1.plot(x_c,cost)
ax1.set_title('cost function')
ax2.plot(x_a,acc)
ax2.set_title('accuracy function')

plt.show()

misci_img= test_x[test_y!=y_pred][:25]
correct_lab= test_y[test_y!=y_pred][:25]
misci_lab=y_pred[test_y != y_pred][:25]


fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
	img=misci_img[i].reshape(28,28)
	ax[i].imshow(img,cmap='viridis', interpolation = 'nearest')
	ax[i].set_title('%d : t:%d p: %d' % (i+1,correct_lab[i],misci_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

