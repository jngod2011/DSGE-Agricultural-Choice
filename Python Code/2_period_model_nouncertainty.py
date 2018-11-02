# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:08:31 2018

@author: rodri
"""

# =============================================================================
# 2 periods model: No uncertainty case
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Set up parameters
θ = np.array([0,1])
ε = np.array([0,1])

#With probabilities equal to 1 we get meaningful results.
pi_θ = np.array([0, 1])
pi_ε = np.array([0, 1])


A = 10
α = 0.4
B = 5
γ = 0.2
m = np.linspace(0,5,100)
ρ = 1
r = 0.02
y_0 = 1
p = 0.5
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



fig,ax = plt.subplots()
ax.plot(m_grid, Ey1(m_grid), label='E(y1)')
ax.plot(m_grid, Ey2(m_grid), label='E(y2)')
ax.plot(m_grid, E_diff_y1(m_grid), label='E(y1_diff)')
ax.plot(m_grid, E_diff_y2(m_grid) , label='E(y2_diff')
ax.legend()
ax.set_xlabel('Input Quantitey')
ax.set_title('xpected value of the production function. No uncertainty case: P(θ=1) = P(ε=1)=1')
plt.show()


#%% Compute optimal values given different levels of initial wealth
policy_a = []
policy_m1 = []
policy_m2 = []
V = []

p= 1
a_0_grid= np.linspace(5,10,50) #for asset levels below 5, there is not an optimal choice (consumption gets negative)
for a_0 in a_0_grid:
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
    print(res.success) #Problem: I get many cases of not success in minimizing
        

fig,ax = plt.subplots()
ax.plot(a_0_grid, np.array(policy_a), label='policy_a')
ax.plot(a_0_grid, np.array(policy_m1), label='policy_m1')
ax.plot(a_0_grid, np.array(policy_m2), label='policy_m2')
ax.legend()
ax.set_xlabel('Initial Assets level')
ax.set_title('Optimal Choices Given Changes in Initial Wealth')
plt.show()


#%% Compute optimal values given changes in input prices
policy_a = []
policy_m1 = []
policy_m2 = []
V = []

y_0 =5
a_0 = 6      #For lower values of initial y,a we do not have solution.
p_grid= np.linspace(0.2,4,50)   #For prices above 2.29 dont have solution
for p in p_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2])
    bounds = ((0,100), (0, 100) , (0, 100))
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    V.append(res.fun)
    print(p)
    print(res.success) 
    
        

fig,ax = plt.subplots()
ax.plot(p_grid, np.array(policy_a), label='policy_a')
ax.plot(p_grid, np.array(policy_m1), label='policy_m1')
ax.plot(p_grid, np.array(policy_m2), label='policy_m2')
ax.legend()
ax.set_xlabel('Farm Input Prices')
ax.set_title('Optimal Choices Given Changes in Inputs Prices')
plt.show()


#%% Compute optimal values given changes in interest rates
policy_a = []
policy_m1 = []
policy_m2 = []
V = []

y_0 =10
a_0 = 5
p=2
r_grid= np.linspace(0,0.2,50)
for r in p_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2])
    bounds = ((0,100), (0, 100) , (0, 100))
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    V.append(res.fun)
    print(res.success) 
    
        

fig,ax = plt.subplots()
ax.plot(r_grid, np.array(policy_a), label='policy_a')
ax.plot(r_grid, np.array(policy_m1), label='policy_m1')
ax.plot(r_grid, np.array(policy_m2), label='policy_m2')
ax.legend()
ax.set_xlabel('Interest Rate')
ax.set_title('Optimal Choices Given Changes in Interest Rate')
plt.show()

#%% Compute optimal values given changes degree of risk aversion
policy_a = []
policy_m1 = []
policy_m2 = []
V = []

y_0 =10
a_0 = 5
p=1
r=0.01
ρ_grid= np.linspace(0,4,50)
for ρ  in ρ_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2])
    bounds = ((0,100), (0, 100) , (0, 100))
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    V.append(res.fun)
    print(res.success) 
        

fig,ax = plt.subplots()
ax.plot(ρ_grid, np.array(policy_a))
ax.plot(ρ_grid, np.array(policy_m1))
ax.plot(ρ_grid, np.array(policy_m2))
ax.set_title('Optimal Choices Given Changes in degree of risk aversion')
plt.show()


