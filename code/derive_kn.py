"""Kerr--Newman--Vaidya: g = g_flat + F l l with F=(2mr-Q^2)/Sigma (ingoing coords).
Checks: d_m g pp = (2r/Sigma)(l.p)^2, d_Q g pp = -(2Q/Sigma)(l.p)^2, l null."""
import sympy as sp
v,r,th=sp.symbols('v r theta',real=True)
m,a,Q=sp.symbols('m a Q',positive=True)
P=sp.Matrix(sp.symbols('pv pr pth pph',real=True))
s,c=sp.sin(th),sp.cos(th)
Sig=r**2+a**2*c**2; D=r**2-2*m*r+a**2+Q**2; W=2*m*r-Q**2
g=sp.zeros(4,4)
g[0,0]=-(1-W/Sig); g[0,1]=g[1,0]=1; g[0,3]=g[3,0]=-a*s**2*W/Sig
g[1,3]=g[3,1]=-a*s**2; g[2,2]=Sig; g[3,3]=((r**2+a**2)**2-D*a**2*s**2)*s**2/Sig
l=sp.Matrix([-1,0,0,a*s**2]); lp=(l.T*P)[0]
q=lambda M:(P.T*M*P)[0]
print('d_m:',sp.simplify(q(g.diff(m))-2*r/Sig*lp**2))
print('d_Q:',sp.simplify(q(g.diff(Q))+2*Q/Sig*lp**2))
gi=sp.simplify(g.inv())
print('l null:',sp.simplify((l.T*gi*l)[0]),' l^mu:',list(sp.simplify(gi*l)))
# flat part independent of m,Q
flat=sp.simplify(g-W/Sig*l*l.T)
print('flat part depends on m,Q:',[sp.simplify(flat.diff(x))==sp.zeros(4,4) for x in (m,Q)])
# Kerr limit check vs Kerr-Vaidya metric
print('Q=0 reduces to Kerr-Vaidya form: g_vv =',sp.simplify(g[0,0].subs(Q,0)))
