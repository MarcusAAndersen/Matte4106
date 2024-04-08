import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

length = 0.75
alfa = 97E-6
temp_left = 25
temp_right = 100
total_time = 1000
dx = 0.05
x_vec = np.linspace(0, length, int(length/dx))
dt = 0.01
t_vec = np.linspace(0, total_time, int(total_time/dt))

u = np.zeros([len(t_vec), len(x_vec)])

# Set initial condition
u[0,:] = 25  
u[0,0] = 25
u[:,-1] = 100

for t in range(1, len(t_vec)):
    for x in range(1, len(x_vec)-1):
        u[t, x] = alfa * (dt / dx**2) * (u[t-1, x+1] - 2*u[t-1, x] + u[t-1, x-1]) + u[t-1, x]
    
    #oppdater venstre grenseverdi
    #u[t, 0] = 0.5 * (u[t, 1] + u[t-1, 0])
    #oppdater høyre grenseverdi
    u[t, 0]=temp_left
    u[t, -1] = temp_right  

X, T = np.meshgrid(x_vec, t_vec)

# Plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, u, cmap='viridis')
ax.set_xlabel('Position')
ax.set_ylabel('Time')
ax.set_zlabel('Temperature')
ax.set_title('Temperature Distribution over Time and Position')
fig.colorbar(surf)
plt.show()