# tungsten temperature resistivity data: https://journals.aps.org/pr/pdf/10.1103/PhysRev.28.202
#measuring resistvitiy of SiC:
#https://nvlpubs.nist.gov/nistpubs/bulletin/07/nbsbulletinv7n1p71_A2b.pdf


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

T_c = np.polyfit(T_temp, T_rho, 2)
T_fit = np.polyval(T_c, T_temp)

axs[0].plot(T_temp, T_fit, color='orange', label='Fit Line')
axs[0].scatter(T_temp,T_rho)
axs[0].set_ylabel('Rho Tungsten (Ohm-m)')
axs[0].set_xlabel('Temperature (K)')


#SiC
high = 0.2e-2
low = 0.01e-2

SiC_rho = [897, 843, 761, 707, 667, 635, 567, 529, 495, 462, 410, 398,407, 388, 316, 295, 163, 145, 122, 102, 77, 54]
SiC_temp = [166, 173, 192, 202, 216, 224, 258, 268, 274, 286, 296, 302, 312, 316, 318, 346, 365, 382, 392, 406, 432, 454]

AL = np.pi*(0.014**2 - 0.01**2)/(4*0.051)

SiC_rho = [rho * AL for rho in SiC_rho]
SiC_temp = [temp + 273 for temp in SiC_temp]

SiC_rho.append((high+low)/2)
SiC_temp.append(1273)

def inverse_model(x, A,B):
    return A/x**B

popt, _ = curve_fit(inverse_model, np.array(SiC_temp), np.array(SiC_rho))
best_fit = inverse_model(np.array(SiC_temp), *popt)

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
      return np.pi*(self.outer**2 - self.inner**2)/4

  def SA(self):
      return np.pi*self.outer*self.length

  def V(self):
      return self.CA()*self.length
   
  def R(self, temperature):
      if self.name == "W":
        rho = np.polyval(T_c, temperature)
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

#constant current power supply
def CC(temp, current, step):
  R = R_tot(temp)
  V = current*R
  P = R*(current**2)
  #T = np.sqrt(np.sqrt((P)/(sigma*e*A_s)))
  T = temp + step*(P - (temp**4)*sigma*e*SiC.SA())/(psi*cp*SiC.V())

  return R,V,P,T

#constant voltage power supply
def CV(temp, voltage, step):
  R = R_tot(temp)
  I = voltage/R
  P = R*(I**2)
  T = temp + step*(P - (temp**4)*sigma*e*SiC.SA())/(psi*cp*SiC.V())

  return R,V,P,T


resistance = []
voltage = []
power = []
temperature = []
current = []
times = []
R_tungsten = []
R_SiC = []

#initalizing
Ti = 300
step = 0.1
scale = 500
time = 0

#heating element properties (these are the ones i can changes with what i have right now)
tungsten.outer
tungsten.length
SiC.length

#current regime (since we need to start with low current then go to high, i thought arctan works very well)
k = 4.5
l = 0.025
m = 10
I = lambda x: k*(np.arctan(l*x - m)+np.pi/2)

T = Ti
while time < scale:
  R_tungsten.append(tungsten.R(T))
  R_SiC.append(SiC.R(T))

  R,V,P,T = CC(T,I(time),step)

  resistance.append(R)
  voltage.append(V)
  power.append(P)
  temperature.append(T)
  times.append(time)
  current.append(I(time))

  time += step

fig, axs = plt.subplots(2,3,figsize=(15, 8))

axs[0,0].plot(times, temperature)
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Temp (K)")

axs[0,1].plot(times, resistance)
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Resistance (Ohm)")

axs[1,0].plot(times, voltage)
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("Voltage (V)")

axs[1,1].plot(times, power)
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("Power (W)")

axs[1,2].plot(times, current)
axs[1,2].set_xlabel("Time (s)")
axs[1,2].set_ylabel("Current (A)")

axs[0,2].plot(times, R_tungsten, label = "Tungsten Resistance")
axs[0,2].plot(times, R_SiC, label = "SiC Resistance")
axs[0,2].set_xlabel("Time (s)")
axs[0,2].set_ylabel("Individual Resistance (R)")
axs[0,2].legend(loc="upper right")

plt.show()
