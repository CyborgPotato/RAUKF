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
    self.g_max_v = bm.Variable(1)
    self.g_max_v.value = self.g_max_v.at[:].set(g_max)
    self.pron = bp.dyn.FullProjDelta(
      pre=pre,
      delay=delay,
      # Event-driven computation
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), self.g_max_v),
      # syn=bp.dyn.Expon(size=post.num, tau=tau),# Exponential synapse
      # out=bp.dyn.COBA(E=E), # COBA network
      post=post
    )

class EINet(bp.DynamicalSystem):
  def __init__(self, noise=True, N=1000, b=0.8, c=0.1):
    super().__init__()
    self.noise = noise
    self.b = b
    ne=int(b*N)
    ni=int((1-b)*N)
    J = 0.1
    K = c*N
    g1 = 100
    Je = J*bm.sqrt(1000/K)
    Ji = (4+g1*bm.sqrt(c/K))*Je
    lif_pars = dict(
      V_rest=0., V_th=20., V_reset=10., tau=20.,
      tau_ref=.5, V_initializer=bp.init.Normal(0., 8.)
    )
    self.E = bp.dyn.LifRef(ne, **lif_pars, method='euler')
    self.I = bp.dyn.LifRef(ni, **lif_pars, method='euler')
    self.E2E = Exponential(self.E, self.E, .55, c, Je,  20., 0.)
    self.E2I = Exponential(self.E, self.I, .55, c, Je,  20., 0.)
    self.I2E = Exponential(self.I, self.E, .55, c, -Ji,  20., -80.)
    self.I2I = Exponential(self.I, self.I, .55, c, -Ji,  20., -80.)
    self.input = bm.Variable(1,batch_axis=0)
    # self.input.value = self.input.at[:].set(500)

    self.lfp = bm.Variable(1,batch_axis=0)
    self.calc_lfp()

  def calc_lfp(self):
    self.lfp.value = self.lfp.at[:].set(self.E.V.mean()*self.b + self.I.V.mean()*(1-self.b))
    # self.lfp.value = self.lfp.at[:].set(
    #   -bm.sum(self.E2E.pron.syn.g) +\
    #    bm.sum(self.I2E.pron.syn.g) +\
    #    bm.sum(self.I2I.pron.syn.g) +\
    #   -bm.sum(self.E2I.pron.syn.g)
    # )

  def update(self):
    t = bp.share['t']
    self.input.value = self.input.at[:].set(
      30*bm.sin(t*bm.pi*2/(1000/2))+30 + 2*t/1000
    )
    # self.input.value = self.input.at[:].set(0)
    # if self.noise:
        # self.input.value +=  -(0.001*(self.input-500)) + bsr.normal(0,10,size=1)
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.calc_lfp()
    self.E(24+self.input)
    self.I(24+self.input)
    return self.E.spike.value, self.I.spike.value

import main as main_
from importlib import reload
reload(main_)
from main import RAUKF, Ukf

def run_with_Q_R(params, inp=0, obs=0):
  t_stop = obs.size/10
  lfp_Q,w_Q = params
  net_kf = RAUKF(
    EINet(False),
    [ # What internal states to track
      r'.*lfp$',
    ],
    [ # What states to estimate
      # r'.*input$',
      r'.*weight$'
    ],
    [ # What our measurement/observation is
      r'.*lfp$',
    ],
    obs
  )
  net_kf.t_stab = 100
  net_kf.x.value = net_kf.x.at[-4:].set(net_kf.x.value[-4:]*1)
  net_kf.Q.value = np.diag(np.array([
      lfp_Q,
      # inp_Q,
      w_Q,
      w_Q,
      w_Q,
      w_Q,
  ],dtype=np.float32))
  net_kf.P.value = np.diag(np.array([
      lfp_Q,
      # inp_Q,
      w_Q,
      w_Q,
      w_Q,
      w_Q,
  ],dtype=np.float32))

  net_kf.R.value = net_kf.R.at[:].set(25)

  kf_run = bp.DSRunner(net_kf, monitors=['net.lfp','net.input'],progress_bar=False)

  _=kf_run.run(t_stop)

  kf_lfp = kf_run.mon['net.lfp']
  kf_input = kf_run.mon['net.input']

  return np.nanmean(np.square(kf_lfp-obs)) +\
    np.mean(np.isnan(kf_lfp-obs)*1e9)

if __name__=='__main__':
  t_stop = 100e3
  dbs_times = np.arange(2500,8000,20)

  net = EINet()

  runner = bp.DSRunner(net, monitors=['lfp','input','E.spike','I.spike'])

  _=runner.run(t_stop)

  observation = runner.mon['lfp']
  input = runner.mon['input']

  obs = observation#+np.random.normal(0,50,size=observation.shape)
  # plt.plot(obs)
  # plt.twinx()
  # plt.plot(input,color='r')
  # plt.show()
  # plt.scatter(*np.nonzero(runner.mon['E.spike']),marker='|')
  # plt.figure()
  # plt.scatter(*np.nonzero(runner.mon['I.spike']),marker='|')
  # plt.show()

  # from multiprocessing import Pool, get_context
  # from functools import partial
  # from itertools import product
  # from tqdm import tqdm

  # sp_ctx = get_context('forkserver')
  # with sp_ctx.Pool(16) as p:
  #   Q_R_range = np.logspace(-9,9,12)
  #   params = list(product(*[Q_R_range]*2))
  #   results = list(
  #       tqdm(p.imap(partial(
  #           run_with_Q_R,
  #           inp=input,
  #           obs=obs
  #       ), params, 4),total=len(params))
  #   )
  #   print(np.sort(results)[:5])
  #   print(np.argsort(results)[:5])
  #   print(np.array(params)[np.argsort(results)[:5]])
  # exit()
  
  # # # plt.plot(input,color='r')
  # # # plt.twinx()
  # # # plt.plot(observation,color='k')
  # # # plt.show()
  net_kf = RAUKF(
    EINet(False),
    [ # What internal states to track
      r'.*lfp$',
    ],
    [ # What states to estimate
      # r'.*input$',
      r'.*weight$'
    ],
    [ # What our measurement/observation is
      r'.*lfp$',
    ],
    obs
  )
  net_kf.t_stab = 0

  # net_kf.R.value = net_kf.R.at[:].set(1e-6)
  # net_kf.robust = False
  # net_kf.lambda0 = 0
  # net_kf.delta0 = 0
  # net_kf.a = 10
  # net_kf.b = 1000
  # net_kf.threshold = .7

  # net_kf.x.value = net_kf.x.at[1].set(11)

  lfp_Q = 1e0
  # inp_Q = 1e16
  we_Q = 1e-3
  wi_Q = 1e-1
  # net_kf.x.value = net_kf.x.at[1].set(net_kf.x.value[1]*1)
  net_kf.x.value = net_kf.x.at[-4:].set(net_kf.x.value[-4:]*0.1)
  net_kf.Q.value = np.diag(np.array([
    lfp_Q,
    # inp_Q,
    we_Q,
    we_Q,
    wi_Q,
    wi_Q,
  ],dtype=np.float32))
  net_kf.P.value = np.diag(np.array([
    lfp_Q,
    # inp_Q,
    we_Q,
    we_Q,
    wi_Q,
    wi_Q,
  ],dtype=np.float32))

  net_kf.R.value = net_kf.R.at[:].set(1e-10)

  net_kf.resample = False
  net_kf.adjust_every = 10
  
  net_kf.robust_after = 0
  net_kf.robust = False
  net_kf.lambda0 = 0.2
  net_kf.delta0 = 0.2
  net_kf.a = 10
  net_kf.b = 10
  net_kf.threshold = 0.45

  kf_run = bp.DSRunner(net_kf, monitors=[
    'net.lfp',
    'net.input',
    'net.E2E.pron.comm.weight',
    'net.E2I.pron.comm.weight',
    'net.I2E.pron.comm.weight',
    'net.I2I.pron.comm.weight',
    'phi',
  ])

  _=kf_run.run(t_stop)

  kf_lfp = kf_run.mon['net.lfp']
  kf_input = kf_run.mon['net.input']

  plt.plot(observation,color='k')
  plt.plot(obs,color='g')
  plt.plot(kf_lfp,linestyle=':',color='r')
  plt.figure()
  plt.plot(kf_run.mon['net.E2E.pron.comm.weight'],color='r')
  plt.hlines(net.E2E.pron.comm.weight.value,0,obs.size,color='r',linestyle='--')
  plt.plot(kf_run.mon['net.E2I.pron.comm.weight'],color='r')
  plt.hlines(net.E2I.pron.comm.weight.value,0,obs.size,color='r',linestyle='--')
  plt.plot(kf_run.mon['net.I2E.pron.comm.weight'],color='g')
  plt.hlines(net.I2E.pron.comm.weight.value,0,obs.size,color='g',linestyle='--')
  plt.plot(kf_run.mon['net.I2I.pron.comm.weight'],color='g')
  plt.hlines(net.I2I.pron.comm.weight.value,0,obs.size,color='g',linestyle='--')
  # plt.figure()
  # plt.plot(input,color='k')
  # plt.plot(kf_input,linestyle=':',color='r',linewidth=5)
  plt.show()
