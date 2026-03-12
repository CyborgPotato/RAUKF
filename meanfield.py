import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
import brainstate.random as bsr

from brainpy.math import cond
import jax.lax as lax
import jax.numpy as jnp
import jax
jax.config.update("jax_logging_level", 'CRITICAL')

import matplotlib.pyplot as plt

import main as main_
from importlib import reload
reload(main_)
from main import RAUKF

class Exponential(bp.Projection):
  def __init__(self, pre, post, delay, prob, g_max, tau, E):
    super().__init__()
    self.g_max_v = bm.Variable(1)
    self.g_max_v.value = self.g_max_v.at[:].set(g_max)
    self.pron = bp.dyn.FullProjAlignPost(
      pre=pre,
      delay=delay,
      # Event-driven computation
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), self.g_max_v),
      syn=bp.dyn.Expon(size=post.num, tau=tau),# Exponential synapse
      out=bp.dyn.COBA(E=E), # COBA network
      post=post
    )

class AdExIF(bp.dyn.neurons.AdExIFLTC):
  def __init__(self, *args, gL=10, **kwargs):
    super().__init__(*args, **kwargs)
    self.gL = bm.Variable(1)
    self.gL.value = self.gL.at[:].set(gL)

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def dV(self, V, t, w, I):
    exp = self.gL * self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dVdt = (-self.gL * (V - self.V_rest) + exp - w + I) / self.tau
    return dVdt

  def update(self, x=None):
    x = 0. if x is None else x
    x = self.sum_current_inputs(self.V.value, init=x)
    return super().update(x)
    
class SNN(bp.DynamicalSystem):
  def __init__(self, noise=True, ne=8000, ni=2000):
    super().__init__()
    self.lfp = bm.Variable(1)
    cmn_args = {
      "V_rest": -63,
      "V_reset": -65,
      "V_T": -50,
      "V_th": 20,
      "tau_w": 500,
      "tau": 200,
      "gL": 10,
      "method": 'euler',
    }
    self.E = AdExIF(
      ne,
      a=4,
      b=60,
      delta_T = 2,
      **cmn_args,
      V_initializer = bp.init.Normal(-65., 0.1),
      # w_initializer = bp.init.Normal(200, 200),
    )
    self.E.tau_ref = 5.
    self.I = AdExIF(
      ni,
      a=0,
      b=0,
      delta_T = 0.5,
      **cmn_args,
      V_initializer = bp.init.Normal(-65., 0.1),
      # w_initializer = bp.init.Normal(200, 200),
    )
    self.I.tau_ref = 5.

    self.E2E = Exponential(self.E, self.E, 2, 0.05, 1.0, 5.,   0.)
    self.E2I = Exponential(self.E, self.I, 2, 0.05, 1.0, 5.,   0.)
    self.I2E = Exponential(self.I, self.E, 2, 0.05, 5.0, 5., -80.)
    self.I2I = Exponential(self.I, self.I, 2, 0.05, 5.0, 5., -80.)

    self.NE = bp.dyn.PoissonInput(self.E2E.pron.syn.g,500,2.5,1)
    self.NI = bp.dyn.PoissonInput(self.E2I.pron.syn.g,500,2.5,1)
    
    self.calc_lfp()

  def calc_lfp(self):
    self.lfp.value = self.lfp.at[:].set(
      -bm.sum(self.E2E.pron.syn.g) +\
       bm.sum(self.I2E.pron.syn.g) +\
       bm.sum(self.I2I.pron.syn.g) +\
      -bm.sum(self.E2I.pron.syn.g)
    )

  def update(self):
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.calc_lfp()
    # t = bp.share.load('t')
    # inp = 800#*bm.exp(-t/10)
    self.E()
    self.I()
    self.NE()
    self.NI()
    return

class MeanField(bp.DynamicalSystem):
  def __init__(self):
    super().__init__()
    
    self.ve = bm.Variable(bm.array([10.]))
    self.vi = bm.Variable(bm.array([10.]))
    self.W  = bm.Variable(1)

    self.Pe = bm.Variable(bm.array([
      -49.8, 5.06, -25, 1.4, -0.41, 10.5, -36, 7.4, 1.2, -40.7
    ]))*1e-3
    self.Pi = bm.Variable(bm.array([
      -51.4, 4.0, -8.3, 0.2, -0.5, 1.4, -14.6, 4.5, 2.8, -15.3
    ]))*1e-3
    # TODO Set these variables
    self.Ke = bm.Variable(bm.array([Ne*p]))
    self.tau_e = bm.Variable(bm.array([5e-3]))
    self.Qe = bm.Variable(bm.array([1.5e-9]))
    self.Ki = bm.Variable(bm.array([Ni*p]))
    self.tau_i = bm.Variable(bm.array([5e-3]))
    self.Qi = bm.Variable(bm.array([5e-9]))
    self.gL = bm.Variable(bm.array([10e-9]))
    self.Ee = bm.Variable(bm.array([0]))
    self.Ei = bm.Variable(bm.array([-80e-3]))
    self.EL = bm.Variable(bm.array([-63e-3]))
    self.Cm = bm.Variable(bm.array([281e-9]))
    #
    self.uv0  = bm.Variable(bm.array([-60e-3]))
    self.stdv0 = bm.Variable(bm.array([0.004e-3]))
    self.tauNv0 = bm.Variable(bm.array([0.5]))
    self.duv  = bm.Variable(bm.array([0.001e-3]))
    self.dstdv = bm.Variable(bm.array([0.006e-3]))
    self.dtauNv = bm.Variable(bm.array([1]))
    self.T = bm.Variable(bm.array([50e-3]))
    self.tau_w = bm.Variable(bm.array([500e-3]))
    self.b = bm.Variable(bm.array([30e-12]))
    self.a = bm.Variable(bm.array([4e-12]))

    self.int_v = bp.odeint(self.dv)
    self.int_W = bp.odeint(self.dW)

  def dv(self, v, t, W, ve, vi,P):
    return (self.F(ve,vi,W,P)-v)/self.T

  def dW(self, W, t, ve, vi):
    return -W/self.tau_w + self.b*ve + self.a*(self.uv(ve,vi,W)-self.EL)

  def uGx(self, v, K, tau, Q):
    return v*K*tau*Q
  
  def uv(self, ve, vi, W):
    uGe = self.uGx(ve, self.Ke, self.tau_e, self.Qe)
    uGi = self.uGx(vi, self.Ki, self.tau_i, self.Qi)
    uG  = uGe + uGi + self.gL
    return (uGe*self.Ee + uGi*self.Ei + self.gL*self.EL - W)/uG

  def stdvx(self, v, K, tau, Q, E, uv):
    uGe = self.uGx(v, K, tau, Q)
    uGi = self.uGx(v, K, tau, Q)
    uG  = uGe + uGi + self.gL
    Us = Q/uG * (E-uv)
    teff_m = self.Cm/uG
    return K*v*bm.square(Us*tau)/(2*(teff_m+tau))

  def stdv(self, ve, vi, W, uv):
    stdve = self.stdvx(ve, self.Ke, self.tau_e, self.Qe, self.Ee, uv)
    stdvi = self.stdvx(vi, self.Ki, self.tau_i, self.Qi, self.Ei, uv)
    return bm.sqrt(stdve + stdvi)

  def tauvx_num(self, v, K, tau, Q, E, uv):
    uGe = self.uGx(v, K, tau, Q)
    uGi = self.uGx(v, K, tau, Q)
    uG  = uGe + uGi + self.gL
    Us = Q/uG * (E-uv)
    return K*v*bm.square(Us*tau)

  def tauvx_den(self, v, K, tau, Q, E, uv):
    uGe = self.uGx(v, K, tau, Q)
    uGi = self.uGx(v, K, tau, Q)
    uG  = uGe + uGi + self.gL
    Us = Q/uG * (E-uv)
    teff_m = self.Cm/uG
    return K*v*bm.square(Us*tau)/(teff_m+tau)
  
  def tauv(self, ve, vi, W, uv):
    tauve_num = self.tauvx_num(ve, self.Ke, self.tau_e, self.Qe, self.Ee, uv)
    tauvi_num = self.tauvx_num(ve, self.Ke, self.tau_e, self.Qe, self.Ee, uv)
    tauve_den = self.tauvx_den(vi, self.Ki, self.tau_i, self.Qi, self.Ei, uv)
    tauvi_den = self.tauvx_den(vi, self.Ki, self.tau_i, self.Qi, self.Ei, uv)
    return (tauve_num+tauvi_num)/(tauve_den+tauvi_den)

  def Veff_th(self, uv, stdv, tauv, P):
    tauNv = tauv * self.gL / self.Cm

    t1 = (uv-self.uv0)/self.duv
    t2 = (stdv-self.stdv0)/self.dstdv
    t3 = (tauNv-self.tauNv0)/self.dtauNv
    t4 = t1*t1
    t5 = t2*t2
    t6 = t3*t3
    t7 = t1*t2
    t8 = t1*t3
    t9 = t2*t3
    
    return P[0] + P[1]*t1 + P[2]*t2 + P[3]*t3 + P[4]*t4 +\
      P[5]*t5 + P[6]*t6 + P[7]*t7 + P[8]*t8 + P[9]*t9

  def F(self, ve, vi, W, P):
    uv   = self.uv  (ve,vi,W)
    stdv = self.stdv(ve,vi,W,uv)
    tauv = self.tauv(ve,vi,W,uv)
    return jax.lax.erfc((self.Veff_th(uv,stdv,tauv,P)-uv)/(bm.sqrt(2)*stdv))/(2*tauv)
  
  def update(self, x=None):
    t = bp.share['t']
    dt = bp.share['dt']
    ve = self.int_v(self.ve,t,self.W,self.ve,self.vi,self.Pe,dt=dt)
    vi = self.int_v(self.vi,t,self.W,self.ve,self.vi,self.Pi,dt=dt)
    W  = self.int_W(self.W,t,self.ve,self.vi,dt=dt)

    self.ve.value = ve
    self.vi.value = vi
    self.W.value = W  
  
if __name__=='__main__':
  t_stop = 2e-1

  mean_net = MeanField()
  mean_run = bp.DSRunner(mean_net, monitors=['ve','vi'])

  _=mean_run.run(t_stop)
  ve = mean_run.mon['ve']
  vi = mean_run.mon['vi']
  plt.plot(ve)
  plt.plot(vi)
  plt.show()
  # net = SNN()

  # snn_run = bp.DSRunner(net, monitors=['lfp','E.spike','I.spike'])

  # _=snn_run.run(t_stop)

  # observation = snn_run.mon['lfp']

  # obs = observation#+np.random.normal(0,50,size=observation.shape)

  # plt.plot(obs)
  # # plt.scatter(*snn_run.mon['E.spike'].nonzero(),marker='|')
  # # plt.scatter(*snn_run.mon['I.spike'].nonzero(),marker='|')
  # plt.show()
