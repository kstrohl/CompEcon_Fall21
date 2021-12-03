# %%
import numpy as np 
import functions as fn 

# Set parameters 
beta = 0.95
sigma = 1.5
alpha = 0.1

# Create grid for state space
lb_z = 0.4 
ub_z = 2.0 
size_z = 200  # Number of grid points
z_grid = np.linspace(lb_z, ub_z, size_z)

# Policy Function Iteration Algorithm
PFtol = 1e-8 
PFdist = 7.0 
PFmaxiter = 500 
phi = lambda w:  w # initial guess at policy function is to eat all cake
PFstore = np.zeros((size_z, PFmaxiter)) # initialize PFstore array
PFiter = 1 
PF_params = (alpha, sigma, beta)
while PFdist > PFtol and PFiter < PFmaxiter:
    PFstore[:, PFiter] = phi(z_grid)
    new_phi = fn.coleman_egm(phi, z_grid, PF_params)
    PFdist = (np.absolute(phi(z_grid) - new_phi(z_grid))).max()
    phi = new_phi
    PFiter += 1
    print('Iteration ', PFiter, ' distance = ', PFdist)

if PFiter < PFmaxiter:
    print('Policy function converged after this many iterations:', PFiter)
else:
    print('Policy function did not converge') 

# %%
import matplotlib.pyplot as plt

# Extract decision rules from solution
optC = phi(z_grid)
optX = fn.new_x(optC, alpha, sigma)
optZ = z_grid - optC - optX


# Visualize output
plt.figure()
fig, ax = plt.subplots()
ax.plot(z_grid[1:], optC[1:], label='"Green" Investment')
legend = ax.legend(loc='upper left', shadow=False)
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)
plt.xlabel('Land Allocation')
plt.ylabel('Optimal "Green" Investment"')
plt.title('Policy Function, consumption - deterministic land investment')
plt.show()
fig.savefig('PS8_PF.png', transparent=False, dpi=800, bbox_inches="tight")

VF = fn.utility(optC, optX, alpha, sigma)
plt.figure()
fig, ax = plt.subplots()
ax.plot(z_grid[1:], VF[1:], label="Social Welfare Function")
legend = ax.legend(loc='upper left', shadow=False)
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)
plt.xlabel('Land Allocation')
plt.ylabel('Utility')
plt.title('Value Function - deterministic land investment')
plt.show()
fig.savefig('PS8_VF.png', transparent=False, dpi=800, bbox_inches="tight")