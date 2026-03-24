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
  def __init__(self, noise=True, N=1000, b=0.8, c=0.05):
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

    self.Je = bm.Variable(1)
    self.Je.value = self.Je.at[:].set(Je)
    self.Ji = bm.Variable(1)
    self.Ji.value = self.Ji.at[:].set(Ji)
    
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
    self.lfp.value = self.lfp.at[:].set(
      self.lfp_trace.mean()
    )

  def update(self):
    self.E2E.pron.comm.weight.value = self.Je
    self.E2I.pron.comm.weight.value = self.Je
    self.I2I.pron.comm.weight.value = self.Ji
    self.I2E.pron.comm.weight.value = self.Ji
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.calc_lfp()
    self.E(self.NE())
    self.I(self.NI())
    return self.E.spike.value, self.I.spike.value

import main as main_
from importlib import reload
reload(main_)
from main import RAUKF, Ukf
import dbs as dbs_
reload(dbs_)
from dbs import DBS

if __name__=='__main__':
  t_stop = 100e3
  dbs_times = np.array([0])#np.arange(2500,8000,10)

  net = EINet()
  # net = DBS(net_,[net_.E],dbs_times,0.05,0.05)

  runner = bp.DSRunner(net, monitors=['lfp','I.spike','E.spike'])

  _=runner.run(t_stop)

  observation = runner.mon['lfp']
  # input = runner.mon['Einput']
  ts = runner.mon['ts']

  obs = observation#+np.random.normal(0,50,size=observation.shape)
  # plt.plot(ts,obs)
  # # plt.twinx()
  # # plt.plot(input,color='r')
  # # plt.show()
  # plt.figure()
  # plt.scatter(*np.nonzero(runner.mon['E.spike']),marker='|')
  # plt.figure()
  # plt.scatter(*np.nonzero(runner.mon['I.spike']),marker='|')
  # plt.show()
  # exit()

  net_kf_ = EINet(False)
  net_kf = RAUKF(
    net_kf_,
    # DBS(net_kf_,[net_kf_.E],dbs_times,0.05,0.05),
    [ # What internal states to track
      # r'.*lfp$',
    ],
    [ # What states to estimate
      # r'.*input$',
      # r'.*Einput$',
      r'.*Je$',
      r'.*Ji$',
    ],
    [ # What our measurement/observation is
      r'.*lfp$',
    ],
    obs
  )
  net_kf.T = 1
  net_kf.t_stab = 500/net_kf.T
  net_kf.resample = True
  net_kf.adjust_every = 0

  # net_kf.R.value = net_kf.R.at[:].set(1e-6)
  # net_kf.robust = False
  # net_kf.lambda0 = 0
  # net_kf.delta0 = 0
  # net_kf.a = 10
  # net_kf.b = 1000
  # net_kf.threshold = .7

  # net_kf.x.value = net_kf.x.at[1].set(11)

  # lfp_Q = 1e-2
  # inp_Q = 1e-15
  we_Q = .25e-10 + 0.125e-10
  wi_Q = .25e-10 + 0.125e-10
  # net_kf.x.value = net_kf.x.at[1].set(net_kf.x.value[1]*1)
  net_kf.x.value = net_kf.x.at[-2:].set(net_kf.x.value[-2:]*0.5)
  net_kf.Q.value = np.diag(np.array([
    # lfp_Q,
    # inp_Q,
    we_Q,
    wi_Q,
  ],dtype=np.float32))
  net_kf.P.value = np.diag(np.array([
    # lfp_Q,
    # inp_Q,
    we_Q,
    wi_Q,
  ],dtype=np.float32))

  net_kf.R.value = net_kf.R.at[:].set(0.01)
  
  net_kf.robust_after = 0
  net_kf.robust = False
  net_kf.lambda0 = 0.2
  net_kf.delta0 = 0.2
  net_kf.a = 10
  net_kf.b = 10
  net_kf.threshold = 0.45

  kf_run = bp.DSRunner(net_kf, monitors=[
    'net.lfp',
    'net.Je',
    'net.Ji',
    'phi',
  ])

  _=kf_run.run(t_stop/net_kf.T)

  kf_lfp = kf_run.mon['net.lfp']
  # kf_input = kf_run.mon['net.Einput']
  kf_ts = kf_run.mon['ts']*net_kf.T

  plt.plot(ts,observation,color='k')
  plt.plot(ts,obs,color='g')
  plt.plot(kf_ts,kf_lfp,linestyle=':',color='r')
  plt.figure()
  plt.plot(kf_ts,kf_run.mon['net.Je'],color='r')
  plt.hlines(net.Je.value,0,ts.max(),color='r',linestyle='--')
  plt.plot(kf_ts,kf_run.mon['net.Ji'],color='g')
  plt.hlines(net.Ji.value,0,ts.max(),color='g',linestyle='--')
  # plt.figure()
  # plt.plot(input,color='k')
  # plt.plot(kf_input,linestyle=':',color='r',linewidth=5)
  plt.show()
