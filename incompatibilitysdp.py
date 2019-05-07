import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GenProjectivePOVM(theta) :
   """GenProjectivePOVM(float theta): generates a povm on a qubit with 
      psi1 = cos(theta)|0>+sin(theta)|1>
      psi2 = sin(theta)|0>-cos(theta)|1>
      M1 = |psi1><psi1| (2x2 Matrix)
      M2 = |psi2><psi2| (2x2 Matrix)
      returns [M1,M2]"""
   psi1 = np.array([np.cos(theta),np.sin(theta)])
   psi2 = np.array([-np.sin(theta),np.cos(theta)])
   M1 = np.outer(psi1,psi1)
   M2 = np.outer(psi2,psi2)
   M = np.zeros(2, dtype = (np.matrix))
   M[0] = M1
   M[1] = M2
   return M

def GenTrinePOVM(theta):
   psi1 = np.array([np.cos(np.pi/3+theta/2),np.sin(np.pi/3+theta/2)])
   psi2 = np.array([np.cos(2*np.pi/3+theta/2),np.sin(2*np.pi/3+theta/2)])
   psi3 = np.array([np.cos(3*np.pi/3+theta/2),np.sin(3*np.pi/3+theta/2)])
   M0 = np.outer(psi1,psi1)
   M1 = np.outer(psi2,psi2)
   M2 = np.outer(psi3,psi3)
   M = np.zeros(3, dtype = (np.matrix))
   M[0] = M0
   M[1] = M1
   M[2] = M2
   return M

def FindTrineIncompatibility(theta, dtheta, p):
   povm1 = GenTrinePOVM(theta)
   povm2 = GenTrinePOVM(theta + dtheta)
   povm1 = QubitDephasingChannel(povm1,p)
   povm2 = QubitDephasingChannel(povm2,p)

   s = cp.Variable(1)
   G = MakeVariables(2,3)

   incompConstraints = [np.sum(G) == s*CreateIdentityArray(2)]
   SetPositivityConstraint(incompConstraints,G,3)

   
   
   

def FindProjectiveIncompatibility(theta ,dtheta, p, printsol = False, dephase = True, returnProb = False) :
   """FindProjectiveIncompatibility(float theta):
   TODO FIX THIS
      calculates the sdp for incompatibility for two POVMs that are projective measurments which are shifted versions of eachother (differing by the angle theta)
      returns float, 1+I_r"""
   povm1 = GenProjectivePOVM(theta)
   povm2 = GenProjectivePOVM(theta+dtheta)
   if dephase:
      povm1 = QubitDephasingChannel(povm1,p)
      povm2 = QubitDephasingChannel(povm2,p)
   else:
      povm1 = QubitRawNoiseChannel(povm1,p)
      povm2 = QubitRawNoiseChannel(povm2,p)

   G00 = cp.Variable((2,2), hermitian = True)
   G01 = cp.Variable((2,2), hermitian = True)
   G10 = cp.Variable((2,2), hermitian = True)
   G11 = cp.Variable((2,2), hermitian = True)
   s = cp.Variable(1)

   incompConstraints = [G00+G01+G10+G11 == s*CreateIdentityArray(2)]
   incompConstraints += [G00 + G01 - povm1[0] >> 0]
   incompConstraints += [G10 + G11 - povm1[1] >> 0]
   incompConstraints += [G10 + G00 - povm2[0] >> 0] 
   incompConstraints += [G11 + G01 - povm2[1] >> 0]
   incompConstraints += [G10 >> 0]
   incompConstraints += [G00 >> 0]
   incompConstraints += [G01 >> 0]
   incompConstraints += [G11 >> 0]
   
   prob = cp.Problem(cp.Minimize(s), constraints = incompConstraints)

   prob.solve()
   if (printsol) :
      print("G00: " + str(G00.value) + ", G01: " + str(G01.value) + ", G10: " + str(G10.value) + ", G11: " + str(G11.value))
   if (returnProb):
      return prob

   return prob.value - 1

def FindIncompatibilityOnQubit(measurearr) :
   #variables = MakeVariables(2,2)
   G00 = cp.Variable((2,2), hermitian = True)
   G01 = cp.Variable((2,2), hermitian = True)
   G10 = cp.Variable((2,2), hermitian = True)
   G11 = cp.Variable((2,2), hermitian = True)
   s = cp.Variable(1)

   incompConstraints = [G00+G01+G10+G11 == s*CreateIdentityArray(2)]
   incompConstraints += [G00 + G01 - measurearr[0,0] >> 0]
   incompConstraints += [G10 + G11 - measurearr[0,1] >> 0]
   incompConstraints += [G10 + G00 - measurearr[1,0] >> 0] 
   incompConstraints += [G11 + G01 - measurearr[1,1] >> 0]
   incompConstraints += [G10 >> 0]
   incompConstraints += [G00 >> 0]
   incompConstraints += [G01 >> 0]
   incompConstraints += [G11 >> 0]

   prob = cp.Problem(cp.Minimize(s), constraints=incompConstraints)
   prob.solve()

   return prob.value - 1

def MaximumDephasingIncompatibility(p) :
   theta0 = np.linspace(-np.pi/4,0, 20)
   Dtheta = np.linspace(0, np.pi/2, 40)
   IR = np.zeros((theta0.size*Dtheta.size,1))
   Full = np.zeros((theta0.size*Dtheta.size,3))

   count = 0
   for theta in theta0 :
      for dtheta in Dtheta :
         r = FindProjectiveIncompatibility(theta, dtheta, p)
         Full[count,:] = np.array([r,theta,theta + dtheta])
         IR[count] = r
         count += 1
   i = np.argmax(IR)
   
   return Full[i,:]

def PlotMaxIncompatibilityWithThetas() :
   var = np.linspace(start = 0, stop = 0.5, num = 50)

   plots = np.zeros((var.size, 3))
   count = 0
   for p in var :
      plots[count,:] = MaximumDephasingIncompatibility(p)
      count += 1

   sol = pd.DataFrame(data = plots, columns = ['I', 'theta', 'dtheta'])

   fig = plt.figure(figsize= [12,8])
   a = plt.axes()
   plt.title("Maximum incompatibility for dephasing on a qubit")
   plt.plot(var,sol['I'])
   a.set(xlabel="p" , ylabel = "Maximum Incompatibility")
   plt.show()
   
   fig = plt.figure(figsize= [12,8])
   a = plt.axes()
   plt.title("Maximum incompatibility thetas for dephasing on a qubit")
   plt.plot(var,sol['theta'], 'b--', label = "Theta1")
   plt.plot(var,sol['dtheta'], 'k--', label = "Theta2")
   a.set(xlabel = "p", ylabel = "theta")
   plt.legend()

   plt.show()

def PlotProjectiveIncompatibility(p):
   var = np.linspace(0, np.pi*(1/2), num = 100)

   irs = np.zeros(100)
   a = 1-np.sin(var)+2*np.sin(var/2)*np.sin(np.pi/4-var/2)
   temp1 = a*(1-np.cos(var))
   sol = a + np.divide(temp1,2*np.sin(var/2)*np.sin(np.pi/4-var/2))-1

   for i in np.arange(100) :
      irs[i] = FindProjectiveIncompatibility(0,var[i],p)
      
   plt.figure(figsize=[12,10])
   plt.axes()
   plt.title("Incompatibility for changing theta")
   plt.plot(var,irs, label = "Numeric")
   plt.plot(var,sol, label = "Exact")
   plt.xlabel("$\\theta$")
   plt.ylabel("$I_r$")
   plt.legend()
   plt.show()


#TODO: Make this not the worst thing ever
def SetDIneq(constarray, variablearray, measurmentarrays) :
   print (variablearray.size)
   for i in np.arange(measurmentarrays[:,0].size) :
      for j in np.arange(measurmentarrays[0,:].size) :
         indexes = [i,j]
         Gtemp = np.zeros((2,2))
         for k in np.arange(variablearray[:,0].size) :
            for m in np.arange(variablearray[0,:].size) :
               if indexes[k] == m :
                  Gtemp += variablearray[m,k]
         constarray += [Gtemp - measurmentarrays[i,j] >> 0]

def SetPositivityConstraint(constarray, variablearray, size) :
   for i in np.arange(size):
      constarray += [variablearray[0,i]>>0]
      constarray += [variablearray[1,i]>>0]

#TODO, make this work for arbitrary number of measures? Maybe input an array with the dimension of the measure arrays (which I guess should be the same size, but still)
def MakeVariables(hilbertsize, measurearrsize) :
   G = np.zeros((2,measurearrsize), dtype = cp.Variable)
   for i in np.arange(measurearrsize) :
      G[0,i] = cp.Variable((hilbertsize,hilbertsize), hermitian = True)
      G[1,i] = cp.Variable((hilbertsize,hilbertsize), hermitian = True)
   return G

def CreateIdentityArray(size):
   I = np.zeros((size,size))
   for i in np.arange(size) :
      I[i,i] = 1

   return I

def QubitDephasingChannel(measurements, p):
   Mstar = measurements
   for i in np.arange(Mstar.size):
         Mstar[i] = (1-p)*Mstar[i] + p*np.matmul(SigmaZ(),np.matmul(Mstar[i],SigmaZ()))
   
   return Mstar

def QubitRawNoiseChannel(measurements, p):
   Mstar = measurements
   for i in np.arange(Mstar.size):
         Mstar[i] = (1-p)*Mstar[i] + p*CreateIdentityArray(2)
   
   return Mstar


def SigmaX():
   x = np.zeros((2,2))
   x[0,1] = 1
   x[1,0] = 1
   return x

def SigmaY():
   y = np.zeros((2,2))
   y[0,1] = np.complex(0,-1)
   y[1,0] = np.complex(0,1)
   return y

def SigmaZ():
   z = np.zeros((2,2))
   z[0,0] = 1
   z[1,1] = -1
   return z