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

import pandas as pd

from joblib import Memory
memory = Memory('sweep_cache',verbose=0)

from multiprocessing import get_context
from tqdm import tqdm

class Exponential(bp.Projection):
  def __init__(self, pre, post, delay, prob, g_max, tau, E):
    super().__init__()
    self.g_max = bm.Variable(1)
    self.g_max.value = self.g_max.at[:].set(g_max)
    self.pron = bp.dyn.FullProjAlignPost(
      pre=pre,
      delay=delay,
      # Event-driven computation
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), self.g_max),
      syn=bp.dyn.Expon(size=post.num, tau=tau),# Exponential synapse
      # out=bp.dyn.COBA(E=E), # COBA network
      out=bp.dyn.CUBA(),
      post=post
    )

class PoissonInput(bp.dyn.PoissonInput):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.weight = bm.Variable(np.array([self.weight]))

class EINet(bp.DynamicalSystem):
  def __init__(self, noise=True, N=1000, b=0.8, c=0.05,_jee=None,_jii=None):
    super().__init__()
    self.noise = noise
    self.b = b
    ne=int(b*N)
    ni=int((1-b)*N)

    K = c*ne
    J = 40
    g = 6
    Je = J/K
    Ji = -g*Je

    if _jee is None:
      Jee = Je
    else:
      Jee = _jee
    if _jii is None:
      Jii = Ji
    else:
      Jii = _jii

    self.Je = bm.Variable(1)
    self.Je.value = self.Je.at[:].set(Je)
    self.Ji = bm.Variable(1)
    self.Ji.value = self.Ji.at[:].set(Ji)
    self.Jee = bm.Variable(1)
    self.Jee.value = self.Jee.at[:].set(Jee)
    self.Jii = bm.Variable(1)
    self.Jii.value = self.Jii.at[:].set(Jii)
    
    lif_pars = dict(
      V_rest=0., V_th=20., V_reset=10.,
      tau_ref=.5, V_initializer=bp.init.Normal(0., 0.),
    )
    self.E = bp.dyn.LifRef(
      ne, tau=2, **lif_pars, method='euler',
    )
    self.I = bp.dyn.LifRef(
      ni, tau=2, **lif_pars, method='euler',
    )
    self.E2E = Exponential(self.E, self.E, 2, c, Je,  2.,  0.)
    self.E2I = Exponential(self.E, self.I, 2, c, Je,  2.,  0.)
    self.I2E = Exponential(self.I, self.E, 2, c, Ji,  2., -80.)
    self.I2I = Exponential(self.I, self.I, 2, c, Ji,  2., -80.)

    # self.NE = PoissonInput(self.E2E.pron.syn.g, int(ne*c), 6., Je)
    # self.NI = PoissonInput(self.E2I.pron.syn.g, int(ne*c), 6., Je)

    self.NE = bp.dyn.OUProcess(ne,0,4,10)
    self.NI = bp.dyn.OUProcess(ni,0,4,10)
    
    self.lfp_trace = bm.Variable(100,batch_axis=0)
    self.lfp = bm.Variable(1,batch_axis=0)
    self.lfp_mean = bm.Variable(1,batch_axis=0)
    self.lfp_max = bm.Variable(1,batch_axis=0)
    self.lfp_min = bm.Variable(1,batch_axis=0)
    self.calc_lfp()

  def calc_lfp(self):
    # self.lfp.value = self.lfp.at[:].set(self.E.V.mean()*self.b + self.I.V.mean()*(1-self.b))
    self.lfp_trace.value = self.lfp_trace.at[:-1].set(
      self.lfp_trace[1:]
    )
    self.lfp_trace.value = self.lfp_trace.at[-1].set(
      bm.sum(self.E2E.pron.syn.g) +\
      bm.sum(self.I2E.pron.syn.g) +\
      bm.sum(self.I2I.pron.syn.g) +\
      bm.sum(self.E2I.pron.syn.g)
    )
    self.lfp_mean.value = self.lfp_mean.at[:].set(
      self.lfp_trace.mean()
    )
    self.lfp.value = self.lfp.at[:].set(
      self.lfp_trace[-1]
    )
    self.lfp_max.value = self.lfp.at[:].set(
      self.lfp_trace.max()
    )
    self.lfp_min.value = self.lfp.at[:].set(
      self.lfp_trace.min()
    )

  def update(self):
    self.E2E.pron.comm.weight.value = self.Jee
    self.E2I.pron.comm.weight.value = self.Je
    self.I2I.pron.comm.weight.value = self.Jii
    self.I2E.pron.comm.weight.value = self.Ji
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.calc_lfp()
    self.E(self.NE())
    self.I(self.NI())
    return self.E.spike.value, self.I.spike.value

from importlib import reload
import main as main_
reload(main_)
from main import RAUKF, Ukf
import dbs as dbs_
reload(dbs_)
from dbs import DBS
import gc
# gc.set_debug(gc.DEBUG_LEAK)

def generate_obs(
    ei_args,t_stop=0,
    dbs_times=np.zeros(1),
    dbs_tgts='E', dbs_pct_aff=0.1, dbs_pct_eff=0.1,
    R_obs=['lfp'], index=0, progress=False,
):
  net_ = EINet(**ei_args)
  if dbs_tgts=='E':
    dbs_tgt=[net_.E]
  elif dbs_tgts=='I':
    dbs_tgt=[net_.I]
  elif dbs_tgts=='EI':
    dbs_tgt=[net_.E,net_.I]
  net = DBS(net_,dbs_tgt,dbs_times,dbs_pct_aff,dbs_pct_eff)

  runner = bp.DSRunner(
    net,
    monitors=[f'net.{R}' for R in R_obs],
    progress_bar=progress
  )

  _=runner.run(t_stop)

  ts = runner.mon['ts']
  obs = np.c_[*[runner.mon[f'net.{R}'] for R in R_obs]]
  # Can't return net directly, only values of (breaks cache)
  return ts,obs,(
    net.net.Je.value,
    net.net.Jee.value,
    net.net.Ji.value,
    net.net.Jii.value
  )

ts,obs,(rJe,rJee,rJi,rJii) = generate_obs({},10000,np.arange(2500,7500,100),progress=True)

plt.plot(ts,obs)
plt.show()
