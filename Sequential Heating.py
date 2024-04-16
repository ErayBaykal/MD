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


sigma = 5.67e-8
e = 0.9
psi = 3210
cp = 700 #my documentation says 670, internet says 730
SiC_ID = 0.01
SiC_OD = 0.014

def CA(ID, OD):
  return jnp.pi*(OD**2 - ID**2)/4

def SA(OD, L):
  return jnp.pi*OD*L

def V(ID, OD, L):
  return CA(ID, OD)*L

def R(temperature, ID, OD, L, name):
  if name == "W":
    rho = quadratic_model(temperature, C, D,E)
  elif name == "SiC":
    rho = inverse_model(temperature, A, B)

  return rho*L/CA(ID, OD)

def R_tot(temp, W_OD, W_L, SiC_L):
  return 1/(1/R(temp, 0, W_OD, W_L, "W") + 1/R(temp, SiC_ID, SiC_OD, SiC_L, "SiC"))


#def CC(temp, current, step): #maybe later can try writing this whole thing as one equation
#  R = R_tot(temp)
#  V = current*R
#  P = R*(current**2)
  #T = np.sqrt(np.sqrt((P)/(sigma*e*A_s)))
#  T = temp + step*(P - (temp**4)*sigma*e*SiC.SA())/(psi*cp*SiC.V())

#  return R,V,P,T


def euler(ti,T, current, step, W_OD, W_L, SiC_L): #maybe later can try writing this whole thing as one equation
  SiC_Vol = V(SiC_ID, SiC_OD, SiC_L)
  SiC_SA = SA(SiC_OD, SiC_L)
  W_SA = SA(W_OD, W_L)
  W_R = R(T, 0, W_OD, W_L, "W")


  res = R_tot(T, W_OD, W_L, SiC_L)
  voltage = current*res
  pow = res*(current**2)
  tf = ti + step*(psi*cp*SiC_Vol/(pow - (T**4)*sigma*e*SiC_SA))

  Pd = W_R*current/W_SA

  return res,voltage,pow,tf


def loss(params):
  W_OD = params[0]
  W_L = params[1]
  SiC_L = params[2]
  lam = params[3]
  z = params[4]

  time = 0
  
  #I = lambda x: k*(jnp.arctan(l*x - m)+jnp.pi/2)
  I = 30
  T = 300
  Tf = 1800
  temp_step = 10
  bad = 0

  while T<Tf:
    T += temp_step
    _,_,Pd,time = euler(time, T, I, temp_step, W_OD, W_L, SiC_L)
    
    if Pd > bad:
      bad = Pd

    print("Max Power Density: ", int(bad))
    print("SiC Length: ",float(SiC_L.item()))
    print("Tungsten Length: ",float(W_L.item()))
    print("Tungsten Thickness: ", float(W_OD.item()))
    print("Time: ", int(time))
    print("Temp: ", int(T))
    print("Lambda: ", int(lam))
    print("z: ", int(z))
    print("")

  return time + lam*(bad-500000) + (time - z**2)**2

#tungsten = heater("W",0, 0.00005, 0.25)
#SiC = heater("SiC", 0.01, 0.014, 0.02)
#k = 4.5
#l = 0.025
#m = 10
#300,1700, 0.1, [0.00005, 0.25, 0.02, 4.5, 0.025, 10]

params = jnp.array([0.0005, 0.4, 0.02, 10, 0])
gradient = jax.grad(loss)

plot = []
x = list(range(1, 11))

def optimize(params, LR=0.00001, N_STEPS=10):
  for _ in range(N_STEPS):
    
    grads = gradient(params)
    plot.append(float(grads[1].item()))
    #params = params - (LR * grads)

    params = params[0:-1] - (LR * grads[0:-1])
    params = jnp.concatenate([params, jnp.array([params[-1] + (LR * grads[-1])])])
    #params = jnp.clip(params, jnp.array([0.00005, 0.1, 0.01, 0 ]), jnp.array([0.002, 10, 0.03, 1000]))
    params = jnp.clip(params, jnp.array([0, 0.1, 0.01, 0 , 0]))

  return params

optimize(params)

plt.plot(x,plot)
plt.show()
