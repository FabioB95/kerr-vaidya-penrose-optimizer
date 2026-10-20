import sympy as sp
r,m,a=sp.symbols('r m a',positive=True)
pv,pr,pph=sp.symbols('p_v p_r p_phi',real=True)
D=r**2-2*m*r+a**2
Pv=(a**2*pv+(a**2+r**2)*pr+a*pph)/r**2
Pr=((a**2+r**2)*pv+D*pr+a*pph)/r**2
Pp=(a*pv+a*pr+pph)/r**2
qa=2*Pp*(2*a*m*Pp+a*r*Pp-2*m*Pv-r*Pr)/r
qm=2/r*(Pv-a*Pp)**2
print('quad_m covariant:',sp.factor(sp.simplify(qm)))
print('quad_a covariant:',sp.factor(sp.simplify(qa)))
# law B: a=chi m, d/dv = mdot (d_m + chi d_a); dE/dtau = -1/2 mdot (qm + chi qa)
chi=sp.symbols('chi')
tot=sp.simplify((qm+ (a/m)*qa))
print('qm + (a/m) qa =',sp.factor(tot))
