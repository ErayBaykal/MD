# tungsten temperature resistivity data: https://journals.aps.org/pr/pdf/10.1103/PhysRev.28.202
#measuring resistvitiy of SiC:
#https://nvlpubs.nist.gov/nistpubs/bulletin/07/nbsbulletinv7n1p71_A2b.pdf


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import jax
import jax.numpy as jnp



fig, axs = plt.subplots(2)


#Tungsten
T_temp = [273, 293, 300]
T_rho= [5.00, 5.49, 5.65, 8.05, 10.56, 13.23, 16.10, 18.99, 21.94, 24.90, 28.10, 31.96, 34.10, 37.18,
        40.35, 43.50, 46.78, 50.00, 53.30, 56.67, 60.00, 63.40, 66.85, 70.38, 73.83, 77.38, 81.00,
        84.69, 88.30, 92.00, 95.74, 99.55, 103.3, 107.2, 111.1, 115.0, 117.1]
T_rho = [rho * 1e-8 for rho in T_rho]

for i in range(len(T_rho)-4):
  T_temp.append((i+4)*100)

T_temp.append(3655)

def quadratic_model(x, C, D, E):
    return C + D*x + E*(x**2)

popt, _ = curve_fit(quadratic_model, jnp.array(T_temp), np.array(T_rho))
best_fit = quadratic_model(jnp.array(T_temp), *popt)

C = popt[0]
D = popt[1]
E = popt[2]

#T_c = np.polyfit(T_temp, T_rho, 2)
#T_fit = np.polyval(T_c, T_temp)

axs[0].plot(T_temp, best_fit, color='orange', label='Fit Line')
axs[0].scatter(T_temp,T_rho)
axs[0].set_ylabel('Rho Tungsten (Ohm-m)')
axs[0].set_xlabel('Temperature (K)')


#SiC
high = 0.2e-2
low = 0.01e-2

SiC_rho = [897, 843, 761, 707, 667, 635, 567, 529, 495, 462, 410, 398,407, 388, 316, 295, 163, 145, 122, 102, 77, 54]
SiC_temp = [166, 173, 192, 202, 216, 224, 258, 268, 274, 286, 296, 302, 312, 316, 318, 346, 365, 382, 392, 406, 432, 454]

AL = jnp.pi*(0.014**2 - 0.01**2)/(4*0.051)

SiC_rho = [rho * AL for rho in SiC_rho]
SiC_temp = [temp + 273 for temp in SiC_temp]

SiC_rho.append((high+low)/2)
SiC_temp.append(1273)

def inverse_model(x, A,B):
    return A/x**B

popt, _ = curve_fit(inverse_model, jnp.array(SiC_temp), np.array(SiC_rho))
best_fit = inverse_model(jnp.array(SiC_temp), *popt)

A = popt[0]
B = popt[1]

axs[1].plot(SiC_temp, best_fit, label='Best Fit Curve', color='red')
axs[1].scatter(SiC_temp,SiC_rho)
axs[1].set_ylabel('Rho SiC (Ohm-m)')
axs[1].set_xlabel('Temperature (K)')

class heater:
  def __init__(self, name, inner, outer, length):
      self.name = name
      self.inner = inner
      self.outer = outer
      self.length = length

  def CA(self):
      return jnp.pi*(self.outer**2 - self.inner**2)/4

  def SA(self):
      return jnp.pi*self.outer*self.length

  def V(self):
      return self.CA()*self.length
   
  def R(self, temperature):
      if self.name == "W":
        rho = quadratic_model(temperature, C, D,E)
      elif self.name == "SiC":
        rho = inverse_model(temperature, A, B)

      return rho*self.length/self.CA()


tungsten = heater("W",0, 0.00005, 0.25)
SiC = heater("SiC", 0.01, 0.014, 0.02)


#basic heating model with only radiative loss - assumed surface area is that of SiC - SiC and tungsten assumed at the same temp
sigma = 5.67e-8
e = 0.9
psi = 3210
cp = 700 #my documentation says 670, internet says 730

def R_tot(temp):
  return 1/(1/tungsten.R(temp) + 1/SiC.R(temp))

def CC(temp, current, step): #maybe later can try writing this whole thing as one equation
  R = R_tot(temp)
  V = current*R
  P = R*(current**2)
  #T = np.sqrt(np.sqrt((P)/(sigma*e*A_s)))
  T = temp + step*(P - (temp**4)*sigma*e*SiC.SA())/(psi*cp*SiC.V())
  print(T)

  return R,V,P,T

def CV(temp, voltage, step):
  R = R_tot(temp)
  I = voltage/R
  P = R*(I**2)
  T = temp + step*(P - (temp**4)*sigma*e*SiC.SA())/(psi*cp*SiC.V())

  return R,V,P,T




def loss(Ti, Tf, euler_step, params):
  tungsten.outer = params[0]
  tungsten.length = params[1]
  SiC.length = params[2]
  k = params[3]
  l = params[4]
  m = params[5]

  time = 0
  I = lambda x: k*(jnp.arctan(l*x - m)+jnp.pi/2)
  T = Ti

  while T<Tf:
    time += euler_step
    T = CC(T, I(time), euler_step)[3]
    
  return time

#tungsten = heater("W",0, 0.00005, 0.25)
#SiC = heater("SiC", 0.01, 0.014, 0.02)
#k = 4.5
#l = 0.025
#m = 10
#300,1700, 0.1, [0.00005, 0.25, 0.02, 4.5, 0.025, 10]

def test(params):
    return loss(301, 1700, 0.1, params)

gradient_fn = jax.grad(test)
print("hmm")
params = jnp.array([0.00005, 0.25, 0.02, 4.5, 0.025, 10])
print(gradient_fn(params))

