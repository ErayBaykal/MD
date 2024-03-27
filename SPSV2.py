#SiC with conductive loss (due to wire connections and alumina cooling)
from sympy import *
from sympy import plot

I = 60 #current
#need to know the voltage limit for the welder so i make sure at 100A it is still enough voltage to pass through the SiC (likely not)
sigma = 5.67e-8
h = 10
Tsur = 296

rho = 0.001 #resistivity of SiC
Do = 0.014 #OD of SiC
Di = 0.01 #ID of SiC
L1 = 0.051 #length of SiC
k1 = 490 #SiC thermal conductivity
e = 0.9 #emmissivity

Ac1 = np.pi*(Do**2 - Di**2)/4
As1 = np.pi*Do*L1 + 2*np.pi*Do**2

#SiC
R = rho * L1/Ac1 #total resistance
V = I*R
Q = R * I**2
T = np.sqrt(np.sqrt((Q)/(sigma*e*S_A))) -273

#electric feedthrough
D2 = 0.0063 #0.25" wire diameter assumed
L2 = 0.051 #2" wire length assumed
k2 = 45 #assumed steel for now

#alumina
D3 = 0.0063*2 #0.25" wire diameter assumed
L3 = 0.051/2 #2" wire length assumed
k3 = 30 #assumed alumina for now
Ac3 = np.pi*(D3**2)/4

#aluminum
W4 = 0.051
L4 = 0.0063*2
k4 = 237

def R_fin(h,D,L,k):
  P = np.pi*D
  Ac = np.pi*(D**2)/4
  As = P*L

  m = np.sqrt(h*P/(k*Ac))
  M = np.sqrt(h*P*k*Ac)
  qf = M*(np.sinh(m*L)+(h/(m*k))*np.cosh(m*L))/(np.cosh(m*L)+(h/(m*k))*np.sinh(m*L))

  R_fin = 1/qf

  return R_fin

Th = Symbol('Th')
hr = e*sigma*(Th**3)

R_wire = R_fin(h,D2,L2,k2)
R_cond = L1/(2*k1*Ac1)
R_rad = 2/(hr*As1)
R_alumina = L3/(k3*Ac3)
R_aluminum = 1/(h*W4**2)
R_punch = R_alumina + R_aluminum

F = (Th-Tsur)/(R_cond + 1/(1/R_punch + 1/R_wire)) + Th/R_rad + - Q/2

Th = int(solve(F, Th)[1])

print("Total Resistance: ", R, "Ohm")
print("Voltage Across: ", int(V), "V")
print("Total Power: ", int(Q), "W")
print("Theoretical Max Temp: ", int(T), "C")
print("Max Temp with Losses: ", Th - 273, "C")
