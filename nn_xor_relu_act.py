import numpy as np
from scipy.special import expit

hidden_size=4
input_size=2
classes=1
X=np.array(([[0, 0],[0, 1],[1, 0],[1, 1]]))
Wh=np.random.randn(hidden_size,input_size)
bh=np.random.randn(hidden_size,1)
Wo=np.random.randn(1,hidden_size)
bo=np.random.randn(classes,1)

T=np.array(([[0],[1],[1],[0]]))

reg=1e-3

def lossFunc():

	loss=0;

	dbo=np.zeros_like(bo)
	dWo=np.zeros_like(Wo)
	hg12=np.zeros_like(bh)
	dbh=np.zeros_like(bh)
	dh12tmp=np.zeros_like(bh)
	dh12=np.zeros_like(bh)
	dWh=np.zeros_like(Wh)
	dhg3=np.zeros_like(bo)

	for i in xrange(4):

		#forward propogation
		h12 = np.dot(Wh, X[i,:][np.newaxis].T) + bh
		#hg=expit(h12)
		#hg= np.tanh(h12)
		#hg = np.maximum(0,h12)
		hg = np.maximum(0.1*h12,h12)
		h3=np.dot(Wo,hg) + bo
		#hg3=expit(h3)
		#hg3 = np.maximum(0,h3)
		hg3 = np.maximum(0.1*h3,h3)
		#hg3=np.tanh(h3)
		y=hg3
		loss+=0.5*(T[i]-y)*(T[i]-y)

		#backward propogation	
		de=-(T[i]-y)
		#dhg3=hg3*(1-hg3)
		#dhg3=(1-hg3*hg3)
		#dhg3=hg3
		dhg3[dhg3 <= 0] = 0.1
		dy=dhg3*de
		dbo+=dy
		dWo+=np.dot(dy,hg.T)
		dh12=np.dot(Wo.T,dy)
		#dh12tmp=hg*(1-hg)*dh12
		dh12tmp=hg
		dh12tmp[dh12tmp <= 0]= 0.1
		dh12tmp*=dh12
		#dh12tmp=(1-hg*hg)*dh12
		dbh+=dh12tmp
		xsub=X[i,:]
		xsub=np.matrix(xsub)	
		dWh+=np.dot(dh12tmp, xsub)
	
	np.clip(dWh,-2,2,dWh)
	np.clip(dWo,-2,2,dWo)
	np.clip(dbh,-2,2,dbh)
	np.clip(dbo,-2,2,dbo)
	return loss,dWh,dWo,dbh,dbo

learning_rate=0.4
for ep in xrange(1000):
	loss,dWh,dWo,dbh,dbo=lossFunc()
	"""
	for param,dparam in zip([Wh,Wo,bh,bo],[dWh,dWo,dbh,dbo]):
		param=param-learning_rate*dparam
	"""
	"""
	print dWh
	print dWo
	print dbh
	print dbo
	print 'before the update'
	print Wh
	print Wo
	print bh
	print bo
	"""
	Wh+=-learning_rate*dWh
	Wo+=-learning_rate*dWo
	bh+=-learning_rate*dbh
	bo+=-learning_rate*dbo
	"""
	print 'after the update'
	print Wh
	print Wo
	print bh
	print bo
	"""
	if(ep%100==0):
		print 'Loss is %f' %(loss)

for i in xrange(4):
		#forward propogation
		h12 = np.dot(Wh, X[i,:][np.newaxis].T) + bh
		#hg=expit(h12)
		#hg=np.tanh(h12)
		#hg=np.maximum(0,h12)
		hg=np.maximum(0.1*h12,h12)
		#print hg
		h3=np.dot(Wo,hg) + bo
		#hg3=expit(h3)
		#hg3=np.tanh(h3)
		#hg3=np.maximum(0,h3)
		hg3=np.maximum(0.1*h3,h3)
		print hg3

