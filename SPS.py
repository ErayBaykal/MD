#https://wiretron.com/wp-content/uploads/2017/04/NiCrTechTips.pdf
#https://www.youtube.com/watch?v=QNS0LrlRHbs

rho =  1.45e-6  #resistivity - kanthal (Ohm m)
N = 35
t = 0.1e-3 #wire thickness
w = 0.4e-3 #wire width
L = 12.35e-3  #tzm to tzm length

I = 120 #current
e = 0.9 #graphite emmissivity
sigma = 5.67e-8 

c_A = N*t*w #heating cross sectional area
s_A_wire = 2*(t+w)*L*N  #surface area of the wire
s_A = np.pi*0.013*0.026 #surface area of the die (to calculate radiative losses)

R = rho * L/(c_A) #total resistance
Pd = (R*I**2)/s_A_wire   #in air power density has to be below 54256.25 W/m² (experimentally found 775862 W/m^2 for kanthal) -> i think i can take this as the number at equilibirum

P = R * I**2

Lreal = L-3e-3 #i'm assuming the power dissipation is condenced to small portion the full length since due to cooling from tzm and steel not all across the wire is hot
Pd_real = P/(s_A_wire*Lreal/L)

T = np.sqrt(np.sqrt((P)/(sigma*e*s_A)))

P_cooling = 10*s_A*(T - 23)  #assumed heating loss due to convection  -> if i end up having to cool TZM i can adjust this value
T_real = np.sqrt(np.sqrt((P-P_cooling)/(sigma*e*s_A)))

print("Kanthal Power Density Limit: ", 780000, "W/m^2" )
print("Theoretical Power Density: " , int(Pd), "W/m^2")
print("Real Power Density: ", int(Pd_real), "W/m^2")
print("Total Power: ", int(P), "W")
print("Theoretical Max Temp: ", int(T), "C")
print("Real Max Temp: ", int(T_real), "C")
