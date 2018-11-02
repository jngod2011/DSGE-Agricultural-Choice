# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:22:52 2018

@author: rodri
"""


# =============================================================================
# 2 periods model: Insurance case
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Set up parameters
θ = np.array([0,1])
ε = np.array([0,1])

#With probabilities equal to 1 we get meaningful results.
pi_θ = np.array([0.2, 0.8])
pi_ε = np.array([0.5, 0.5])


A = 2
α = 0.4
B = 2
γ = 0.2
m_max=80
m_grid = np.linspace(0,m_max,100)
ρ = 1
r = 0.05
y_0 = 1
β = 0.95
z_bound = A*m_max**α
a_0 = 5
p = 0.1
q=pi_θ[0]*β

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
policy_z = []
V = []

a_0 = 5
p = 0.05
#q=pi_θ[0]
q=0.07

a_0_grid= np.linspace(0,80,50) #for asset levels below 5, there is not an optimal choice (consumption gets negative)
for a_0 in a_0_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a -q*z) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((0,100), (0, 100) , (0, 100), (0, z_bound)) #No-borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success) #Problem: I get many cases of not success in minimizing
        

fig,ax = plt.subplots()
ax.plot(a_0_grid, np.array(policy_a), label='policy_a')
ax.plot(a_0_grid, np.array(policy_m1), label='policy_m1')
ax.plot(a_0_grid, np.array(policy_m2), label='policy_m2')
ax.plot(a_0_grid, np.array(policy_z), label='policy_z')
ax.legend()
ax.set_xlabel('Initial Assets level')
ax.set_title('Optimal Choices Given Changes in Initial Wealth')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_wealth_insurance.png')


#%% Compute optimal values given changes in input prices
policy_a = []
policy_m1 = []
policy_m2 = []
policy_z = []
V = []

a_0 = 5
p = 0.05
#q=pi_θ[0]
q=0.07

p_grid= np.linspace(0.05,1,50)   #For prices above 2.29 dont have solution
for p in p_grid:
    def U(x):
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a -q*z) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((0,100), (0, 100) , (0, 100), (0, z_bound)) #No-borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success) #Problem: I get many cases of not success in minimizing
    
        

fig,ax = plt.subplots()
ax.plot(p_grid, np.array(policy_a), label='policy_a')
ax.plot(p_grid, np.array(policy_m1), label='policy_m1')
ax.plot(p_grid, np.array(policy_m2), label='policy_m2')
ax.plot(p_grid, np.array(policy_z), label='policy_z')
ax.legend()
ax.set_xlabel('Farm Input Prices')
ax.set_title('Optimal Choices Given Changes in Inputs Prices')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_inputprices_insurance.png')

#%%Optimal choices given changes in insurance price
policy_a = []
policy_m1 = []
policy_m2 = []
policy_z = []
V = []

a_0 = 5
p = 0.05
q=0.07

q_grid= np.linspace(0,0.8,50)   #For prices above 2.29 dont have solution
for q in q_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a -q*z) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((0,100), (0, 100) , (0, 100), (0, z_bound)) #No-borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success) #Problem: I get many cases of not success in minimizing
    
        

fig,ax = plt.subplots()
ax.plot(q_grid, np.array(policy_a), label='policy_a')
ax.plot(q_grid, np.array(policy_m1), label='policy_m1')
ax.plot(q_grid, np.array(policy_m2), label='policy_m2')
ax.plot(q_grid, np.array(policy_z), label='policy_z')
ax.legend()
ax.set_xlabel('insurance Price')
ax.set_title('Optimal Choices Given Changes in Insurance Price')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_insuranceprice_insurance.png')


#%% Optimal policies given changes in borrowing constraint
policy_a = []
policy_m1 = []
policy_m2 = []
policy_z = []
V = []

a_0 = 5
p = 0.05
q=0.07

#problem with nans
b_grid= np.linspace(-100,0,50)   
for b in b_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(max(y_0 +a_0 -p*(m1 + m2) -a -q*z, 0.0001)) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((b,100), (0, 100) , (0, 100), (0, z_bound)) 
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success) 
    
        

fig,ax = plt.subplots()
ax.plot(b_grid, np.array(policy_a), label='policy_a')
ax.plot(b_grid, np.array(policy_m1), label='policy_m1')
ax.plot(b_grid, np.array(policy_m2), label='policy_m2')
ax.plot(b_grid, np.array(policy_z), label='policy_z')
ax.legend()
ax.set_xlabel('Borrowing constraint')
ax.set_title('Optimal Choices Given Changes in Borrowing Constraint')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_borrowcons_insurance.png')




#%% Compute optimal values given changes in interest rates
policy_a = []
policy_m1 = []
policy_m2 = []
policy_z = []
V = []

a_0 = 5
p = 0.05
q=0.07

r_grid= np.linspace(0,0.2,50)
for r in p_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(y_0 +a_0 -p*(m1 + m2) -a -q*z) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((0,100), (0, 100) , (0, 100), (0, z_bound)) #No-borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success)  
    
        

fig,ax = plt.subplots()
ax.plot(r_grid, np.array(policy_a), label='policy_a')
ax.plot(r_grid, np.array(policy_m1), label='policy_m1')
ax.plot(r_grid, np.array(policy_m2), label='policy_m2')
ax.plot(r_grid, np.array(policy_z), label='policy_z')
ax.legend()
ax.set_xlabel('Interest Rate')
ax.set_title('Optimal Choices Given Changes in Interest Rate')
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_interestrate_insurance.png')


#%% Compute optimal values given changes degree of risk aversion
policy_a = []
policy_m1 = []
policy_m2 = []
policy_z = []
V = []

a_0 = 5
p = 0.1
q=0.07
r=0.05

ρ_grid= np.linspace(0.05,4,50)
for ρ  in ρ_grid:
    def U(x):
    # *args pi_θ=pi_θ , pi_ε = pi_ε, θ = θ ,ε = ε, u=u, r=r, β = β
        a = x[0]
        m1= x[1]
        m2= x[2]
        z = x[3]
        #return - (u(max(y_0 +a_0 -p*(m1 + m2) -a, 0.00001)) +β*u(max(A*m1**α +B*m2**γ +(1+r)*a,0.00001))*pi_θ[1]*pi_ε[1] +β*u(max(B*m2**γ +(1+r)*a,0))*pi_θ[0]*pi_ε[1] + β*u(max((1+r)*a,0.00001))*pi_θ[0]*pi_ε[0] + β*u(max((1+r)*a,0.00001))*pi_θ[1]*pi_ε[0])
        return - (u(max(y_0 +a_0 -p*(m1 + m2) -a -q*z,0.0001)) +β*u(A*m1**α +B*m2**γ +(1+r)*a)*pi_θ[1]*pi_ε[1] +β*u(B*m2**γ +z +(1+r)*a)*pi_θ[0]*pi_ε[1] + β*u((1+r)*a +z)*pi_θ[0]*pi_ε[0] + β*u((1+r)*a)*pi_θ[1]*pi_ε[0])

    x0 = np.array([2,2,2,5])
    bounds = ((0,100), (0, 100) , (0, 100), (0, z_bound)) #No-borrowing constraint
    res = minimize(U, x0, method='SLSQP' , bounds=bounds)
    policy_a.append(res.x[0])
    policy_m1.append(res.x[1])
    policy_m2.append(res.x[2])
    policy_z.append(res.x[3])
    V.append(res.fun)
    print(res.success) #Problem: I get many cases of not success in minimizing
        

fig,ax = plt.subplots()
ax.plot(ρ_grid, np.array(policy_a), label='policy_a')
ax.plot(ρ_grid, np.array(policy_m1), label='policy_m1')
ax.plot(ρ_grid, np.array(policy_m2), label='policy_m2')
ax.plot(ρ_grid, np.array(policy_z), label='policy_z')
ax.set_title('Optimal Choices Given Changes in degree of risk aversion')
ax.legend()
ax.set_xlabel('Degree of risk aversion (CRRA). a_0=80(rich)')
plt.show()

fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/2period_riskaversion_rich_insurance.png')
