import numpy as np
import random

stepsize= 1 
imgsize = 4
clssize = 3

class network:
  def __init__(self,layers,k):
    for i in range(layers-2):
      lay[i] =  layer(k[i],k[i+1])


class layer:
  def __init__(self,x,y): # x = ImageVector Size ... y = Number of Classes
    self.x = x
    self.y = y
    self.m = np.random.random((y,x))

  def disp(self):
    print(self.m)
  
  def classify(self,img):
    f = sigma(self.m.dot(img)) #Sigma is the Non Linearity Term
    return f

  def update(self,val,chain):
    chain = sigmaprime(val,chain)  
    #Matrix Back Propagation
    for i in range(self.y):
      for j in range(self.x):
        self.m[i][j] -= stepsize*self.m[i][j]*chain[i]
    
#Class Definition Ends  

#SoftMax Squash
def softmax(f):
  f -= np.max(f) #Scaling For numerical stability
  p = np.exp(f)/np.sum(np.exp(f))
  return p

#Loss Function 
def loss(p,correct): #Make Sure p is after the Softmax
  return -np.log(p[correct])

#The Gradient Function
def lossgrad(p,correct):  #Make Sure p is after the Softmax
  df = p
  df[correct] = 1 - p[correct]
  return df

# The Non Linearity Functions
def sigmaprime(x,chain):
  return 1*chain;

def sigma(x):
  return x

#l = layer(imgsize,clssize)
#img = np.random.random((imgsize))


#l.disp()
#l.update([1,1,1],[1,1,1])
#l.disp()
