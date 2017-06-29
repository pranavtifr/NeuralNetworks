import numpy as np
import random

stepsize= 1 
imgsize = 3
clssize = 2
datasize = 2000
testsize = 10
regstren = 1e-3
totalepoch = 10
# The Image class containing the image vector and the correct class of the image
class images:
  def __init__(self,v,correct):
    self.v = v
    self.ans = correct

#Class to have the Entire network made up of the Layers
class network:
  """ 
  The Network needs to be Initialized in this way
  eg,
  net = network(#of_layers,array_containing_the_neurons_in_every_layer)
  """
  def __init__(self,Linp,k):
    self.lay = []
    self.layers = Linp
    for i in range(self.layers-1):
      self.lay.append(layer(k[i],k[i+1]))

  def Nclassify(self,img):
    k = img
    temp = img
    for i in range(self.layers-1):
      temp = self.lay[i].Lclassify(k) 
      k = temp
    return k

  def train(self,img):
    print('In Train')
    loss = 0
    for i in range(datasize):
      f = []
      k = img[i]
      for l in range(self.layers-1):
        k = self.lay[l].Lclassify(k)
        disp(k)
        f.append(k)
      print('Forward')
      f[0].v = [ 0.02423333, -0.31737436]
      disp(f[0])
      print('__')
      p = softmax(f[self.layers-2].v)
      print(p)
      disp(f[0])
      print("_______")
      loss += lossval(p,img[i].ans)
      df = lossgrad(p,img[i].ans) #Found Gradient on the Final Scores
      for m in reversed(range(self.layers-1)):
        disp(f[m])
      print('Backward')
      #BackPropagation
      dh = df
      for m in reversed(range(self.layers-1)):
        w = self.lay[m]
        reg = self.lay[m].m
        self.lay[m].update(f[m].v,dh)
        dh = sigmaprime(f[m].v,dh)
        for j in range(self.lay[m].y):
          sum = 0
          for i in range(self.lay[m].x):
            sum += dh[i]*w[j][i]
          dh[j] = sum
        self.lay[m].m -= reg*regstren #Regularisation Gradient
      #Regularization Loss in Every Matrix
    print(loss/datasize) # Prints loss after every epoch. 

  def test(self,allimgs):
    count = 0
    for img in allimgs:
      #print("Test" ,i)
      f=self.Nclassify(img)
      #print(f)
      p = softmax(f.v)
      disp(img)
      print(p)
      if p[img.ans] > 0.90:
        count += 1
    print('Test Results')
    print(count,testsize)



class layer:
  def __init__(self,x,y): # x = ImageVector Size ... y = Number of Classes
    self.x = x
    self.y = y
    self.m = np.random.random((y,x))
    self.b = np.random.random((y))
  
  def disp(self):
    print(self.m)
  
  def Lclassify(self,img):
    f = images(sigma(self.m.dot(img.v)),img.ans) #Sigma is the Non Linearity Term
    return f

  def update(self,val,chain):
    k = sigmaprime(val,chain)  
    #Matrix Back Propagation
    for i in range(self.y):
      for j in range(self.x):
        self.m[i][j] -= stepsize*k[i]*val[j] #This is Steepest Gradient Try others later 
    self.b -= chain 

#Class Definition Ends  

#SoftMax Squash
def softmax(f):
  f -= np.max(f) #Scaling For numerical stability
  p = np.exp(f)/np.sum(np.exp(f))
  return p

#Loss Function 
def lossval(p,correct): #Make Sure p is after the Softmax
  return -np.log(p[int(correct)])

#The Gradient Function
def lossgrad(p,correct):  #Make Sure p is after the Softmax
  df = p
  df[int(correct)] = p[int(correct)] -1
  return df

# The Non Linearity Functions
def sigmaprime(x,chain):
  return 1*chain;

def sigma(x):
  return x

def disp(img):
    print(img.v,img.ans)

#Image Generation

def gendata(size):
  l = []
  for i in range(size):
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    
    if 2*x + 3*y + 5*z > 0: #THe Classification Condition
      ans = 0
    else:
      ans = 1
    l.append(images([x,y,z],ans))
  return l

data = gendata(datasize)
check = gendata(testsize)
net = network(2,[imgsize,clssize])
print('This is the Image')
disp(data[0])
for epoch in range(totalepoch):
  print('Epoch',epoch)
  net.train(data)
  print('_____________________________')
net.test(check)
print("This Works")

# l = []
# f = [1,2,3,4,5]
# l.append(f)
# print(softmax(l[0]))
# print(l[0])

