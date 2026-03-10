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
  pass
  
if __name__=='__main__':
  t_stop = .5e3

  net = SNN()

  snn_run = bp.DSRunner(net, monitors=['lfp'])#,'E.spike','I.spike'])

  _=snn_run.run(t_stop)

  observation = snn_run.mon['lfp']

  obs = observation#+np.random.normal(0,50,size=observation.shape)

  plt.plot(obs)
  # plt.scatter(*snn_run.mon['E.spike'].nonzero(),marker='|')
  # plt.scatter(*snn_run.mon['I.spike'].nonzero(),marker='|')
  plt.show()
