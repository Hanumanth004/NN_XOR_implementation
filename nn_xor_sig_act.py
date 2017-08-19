import numpy as np
from scipy.special import expit

hidden_size=2
input_size=2
classes=1
X=np.array(([[0, 0],[0, 1],[1, 0],[1, 1]]))
Wh=np.random.randn(hidden_size,input_size)
bh=np.random.randn(hidden_size,1)
Wo=np.random.randn(1,hidden_size)
bo=np.random.randn(classes,1)

T=np.array(([[0],[1],[1],[0]]))


def lossFunc():

	loss=0;

	dbo=np.zeros_like(bo)
	dWo=np.zeros_like(Wo)
	hg12=np.zeros_like(bh)
	dbh=np.zeros_like(bh)
	dh12tmp=np.zeros_like(bh)
	dh12=np.zeros_like(bh)
	dWh=np.zeros_like(Wh)

	for i in xrange(4):

		#forward propogation
		h12 = np.dot(Wh, X[i,:][np.newaxis].T) + bh
		hg=expit(h12)
		h3=np.dot(Wo,hg) + bo
		hg3=expit(h3)
		y=hg3
		loss+=0.5*(T[i]-y)*(T[i]-y)

		#backward propogation	
		de=-(T[i]-y)
		dhg3=hg3*(1-hg3)
		dy=dhg3*de
		dbo+=dy
		dWo+=np.dot(dy,hg.T)
		dh12=np.dot(Wo.T,dy)
		dh12tmp=hg*(1-hg)*dh12
		dbh+=dh12tmp
		xsub=X[i,:]
		xsub=np.matrix(xsub)	
		dWh+=np.dot(dh12tmp, xsub)

	#np.clip(dWh,-2,2,dWh)
	#np.clip(dWo,-2,2,dWo)
	#np.clip(dbh,-2,2,dbh)
	#np.clip(dbo,-2,2,dbo)
	return loss,dWh,dWo,dbh,dbo

learning_rate=0.2
f2=open("./log_data/sig_0.2.dat",'w')

for ep in xrange(2000):
	loss,dWh,dWo,dbh,dbo=lossFunc()
	
	Wh+=-learning_rate*dWh
	Wo+=-learning_rate*dWo
	bh+=-learning_rate*dbh
	bo+=-learning_rate*dbo

	if(ep%100==0):
		print 'Loss is %f' %(loss)
		
		f2.write(str(ep))
		f2.write("\t")
		tmp=str(loss).strip('[]')
		f2.write(str(tmp))
		f2.write("\n")
		

f2.close()

for i in xrange(4):
		#forward propogation
		h12 = np.dot(Wh, X[i,:][np.newaxis].T) + bh
		hg=expit(h12)
		#print hg
		h3=np.dot(Wo,hg) + bo
		hg3=expit(h3)
		print hg3

