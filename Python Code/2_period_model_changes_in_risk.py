# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:31:16 2018

@author: rodri
"""



# =============================================================================
# 2 periods model
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Set up parameters
θ = np.array([0,1])
ε = np.array([0,1])

#With probabilities equal to 1 we get meaningful results.
pi_θ = np.array([0.2, 0.5])
pi_ε = np.array([0.5, 0.5])


A = 10
α = 0.4
B = 5
γ = 0.2
m = np.linspace(0,5,100)
ρ = 1
r = 0.01
y_0 = 10
p = 0.1
β = 0.95

#Set-up functions

y1 = lambda m1,ε,θ: ε*θ*A*m1**α
y2 = lambda m2,ε: ε*B*m2**γ
y = lambda y1, y2: y1 +y2
diff_y1 = lambda m1,ε,θ: α*ε*θ*A*m1**(α-1)
diff_y2 = lambda m2,ε: γ*ε*B*m2**(γ-1)

Ey1 = lambda m1: pi_θ[1]*pi_ε[1]*A*m1**α
Ey2 = lambda m2: pi_ε[1]*B*m2**γ

E_diff_y1 =  lambda m1: pi_θ[1]*pi_ε[1]*α*A*m1**(α-1)
E_diff_y2 =  lambda m2: pi_ε[1]*γ*B*m2**(γ-1)

def u(c):
    if ρ==1:
        return np.log(c)
    else:
        return (c**(1-ρ) -1) / (1-ρ)


#%%Check that production funcions accomplish the conditions of expected return and expected marginal returns.
m_grid = np.linspace(0.5,20,100)



pi_θ_grid = [np.array([0,1]), np.array([0.1,0.9]), np.array([0.2,0.8]), np.array([0.3,0.7]), np.array([0.4,0.6]), np.array([0.5,0.5]), np.array([0.6,0.4]), np.array([0.7,0.3]), np.array([0.8, 0.2]), np.array([0.9,0.1]), np. array([1, 0]) ]

for  pi_θ in pi_θ_grid:
    Ey1 = lambda m1: pi_θ[1]*pi_ε[1]*A*m1**α
    Ey2 = lambda m2: pi_ε[1]*B*m2**γ
    E_diff_y1 =  lambda m1: pi_θ[1]*pi_ε[1]*α*A*m1**(α-1)
    E_diff_y2 =  lambda m2: pi_ε[1]*γ*B*m2**(γ-1)
    fig,ax = plt.subplots()
    ax.plot(m_grid, Ey1(m_grid), label='E(y1)')
    ax.plot(m_grid, Ey2(m_grid), label='E(y2)')
    ax.plot(m_grid, E_diff_y1(m_grid), label='E(y1_diff)')
    ax.plot(m_grid, E_diff_y2(m_grid) , label='E(y2_diff')
    ax.legend()
    ax.set_xlabel('Input Quantity')
    ax.set_title('Expected value of the production function. Uncertainty case')
    plt.show()


#%% Compute optimal values given different levels of initial wealth
policy_a = []
policy_m1 = []
policy_m2 = []
V = []

a_0 = 50

for  pi_θ in pi_θ_grid:
    print(pi_θ[1])
    
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2])
    bounds = ((0,100), (0, 100) , (0, 100)) #No borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    V.append(res.fun)
    print(res.success) 
        
grid = np.linspace(0,1,11)
fig,ax = plt.subplots()
ax.plot(grid, np.array(policy_a), label='policy_a')
ax.plot(grid, np.array(policy_m1), label='policy_m1')
ax.plot(grid, np.array(policy_m2), label='policy_m2')
ax.legend()
ax.set_xlabel(' pi_θ grid')
ax.set_title('Optimal Choices Given Changes in pi_θ')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_optimality_changes_in_theta.png')


