import numpy as np
import random

stepsize= 1 

class layer:
  def __init__(self,x,y):
    self.x = x
    self.y = y
    self.m = np.random.random((y,x))

  def disp(self):
    print(self.m)

  def update(self,val,chain):
    chain = sigmaprime(val,chain)  
    #Matrix Back Propagation
    for i in range(self.y):
      for j in range(self.x):
        self.m[i][j] -= stepsize*self.m[i][j]*chain[i]
    
#Class Definition Ends  

def sigmaprime(x,chain):
  return 1*chain;

def sigma(x):
  return x

l = layer(2,3)
l.disp()
l.update([1,1,1],[1,1,1])
l.disp()
