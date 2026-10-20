import numpy as np, time, kvcore as K
P=np.array([0.,1e-4,0.05,0.9,1.])
pr=K.pr_from_shell(10.0,1.0,0.9,-1.0,2.0,1.0,'in')
y0=np.array([0.,10.,0.,-1.,pr,2.0])
K.integrate(y0,P,1e5,200.,True,0.05)
t=time.time()
for i in range(20):
    y,tau,st,rmin,tr,nt=K.integrate(y0,P,1e5,200.,True,0.05)
print('incoming',st,y[1],(time.time()-t)/20)
p2,p3=K.decay(y,P,1.0,0.3)
for p in (p2,p3):
    yy=np.concatenate([y[:3],p])
    t=time.time()
    yf,tau,st,rmin,tr,nt=K.integrate(yy,P,1e6,200.,False,0.)
    print(st,-p[0],-yf[3],yf[0],time.time()-t)
