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
    self.lfp.value = self.lfp.at[:].set(
      self.lfp_trace.mean()
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

import main as main_
from importlib import reload
reload(main_)
from main import RAUKF, Ukf
import dbs as dbs_
reload(dbs_)
from dbs import DBS

def fit_lfp_obs(
    obs=0,t_stop=0,b=0.8,c=0.05,T=1,t_stab=0,progress=False,
    Je_scale=1,Ji_scale=1,
):
  net_kf_ = EINet(b=b,c=c)
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
      r'.*Jee$',
      r'.*Ji$',
      r'.*Jii$',
    ],
    [ # What our measurement/observation is
      r'.*lfp$',
      r'.*lfp_max$',
      r'.*lfp_min$',
    ],
    obs
  )
  net_kf.T = T
  net_kf.t_stab = t_stab/net_kf.T
  net_kf.resample = True
  net_kf.adjust_every = 0

  # net_kf.R.value = net_kf.R.at[:].set(1e-6)
  net_kf.robust = False
  net_kf.lambda0 = 0
  net_kf.delta0 = 0
  net_kf.a = 5
  net_kf.b = 5
  net_kf.threshold = 10000000

  # net_kf.x.value = net_kf.x.at[1].set(11)

  # lfp_Q = 1e-2
  # inp_Q = 1e-15
  we_Q = .25e-10 + 0.125e-10
  wi_Q = .25e-10 + 0.125e-10
  # net_kf.x.value = net_kf.x.at[1].set(net_kf.x.value[1]*1)
  net_kf.x.value = net_kf.x.at[-4].set(net_kf.x.value[-4]*Je_scale)
  net_kf.x.value = net_kf.x.at[-3].set(net_kf.x.value[-3]*Je_scale)
  net_kf.x.value = net_kf.x.at[-2].set(net_kf.x.value[-2]*Ji_scale)
  net_kf.x.value = net_kf.x.at[-1].set(net_kf.x.value[-1]*Ji_scale)
  net_kf.Q.value = np.diag(np.array([
    # lfp_Q,
    # inp_Q,
    we_Q,
    we_Q,
    wi_Q,
    wi_Q,
  ],dtype=np.float32))
  net_kf.P.value = np.diag(np.array([
    # lfp_Q,
    # inp_Q,
    we_Q,
    we_Q,
    wi_Q,
    wi_Q,
  ],dtype=np.float32))

  net_kf.R.value = net_kf.R.at[0,0].set(0.01)
  net_kf.R.value = net_kf.R.at[1,1].set(0.01)
  net_kf.R.value = net_kf.R.at[2,2].set(0.01)

  kf_run = bp.DSRunner(
    net_kf, monitors=[
      'net.lfp',
      'net.Jee',
      'net.Je',
      'net.Jii',
      'net.Ji',
      'P',
      'phi',
    ],
    progress_bar=progress,
  )

  _=kf_run.run(t_stop/net_kf.T)
  return kf_run
  
def rmse(a,tgt):
  return np.sqrt(np.mean(np.square(a-tgt)))

if __name__=='__main__':
  b_range = np.linspace(0,1,11)[1:-1]
  # c_range = np.linspace(0,1,11)[1:-1]
  
  t_stop = 10e3
  dbs_times = np.array([0])#np.arange(2500,8000,10)

  obses = []

  b_range = np.array([0.8])

  for b in b_range:
    net = EINet(b=b)
    # net = DBS(net_,[net_.E],dbs_times,0.05,0.05)

    runner = bp.DSRunner(
      net,
      monitors=[
        'lfp','lfp_max','lfp_min',
        'I.spike','E.spike'
      ],
    )

    _=runner.run(t_stop)

    observation = runner.mon['lfp']
    ts = runner.mon['ts']

    obs = np.c_[runner.mon['lfp'],runner.mon['lfp_max'],runner.mon['lfp_min']]
    obses.append(obs)

  from multiprocess import get_context
  from tqdm import tqdm
  from itertools import product, combinations
  from functools import partial
  ctx = get_context('spawn')
  
  kf_T = 1
  t_stab = 100

  with ctx.Pool(8) as p:
    _init_scales = np.logspace(-3,3,13,base=2)
    init_scales = list(product(*[_init_scales]*2))

    obs_idxs = np.arange(b_range.size)
    init_scales = [[1,1]+list(x) for x in product(b_range,obs_idxs)]

    def __run(params,Je=0,Ji=0,obses=[],**kwargs):
      Je_scale,Ji_scale,b,obs_idx = params
      obs = obses[obs_idx]
      kf_run = fit_lfp_obs(
        Je_scale=Je_scale,Ji_scale=Ji_scale,obs=obs,b=b,
        **kwargs,
      )
      Je_rmse = rmse(kf_run.mon['net.Je'],Je)
      Ji_rmse = rmse(kf_run.mon['net.Ji'],Ji)
      return Je_rmse, Ji_rmse
    run = partial(
      __run,obses=obses,t_stop=t_stop,T=kf_T,t_stab=t_stab,
      Je=net.Je.value,Ji=net.Ji.value,
    )

    n_sim = len(init_scales)
    runs = p.imap(run,init_scales,chunksize=max(1,n_sim//8//4))
    rmses = list(tqdm(runs,total=n_sim))
  rmses = np.array(rmses)
  init_scales = np.array(init_scales)
  np.savez('rmse_dat_off_model.npz',rmses=rmses,args=init_scales)

  # kf_lfp = kf_run.mon['net.lfp']
  # # kf_input = kf_run.mon['net.Einput']
  # kf_ts = kf_run.mon['ts']*kf_T
  # stab_mask = kf_ts>=t_stab

  # plt.plot(ts,observation,color='k')
  # plt.plot(ts,obs,color='g')
  # plt.plot(kf_ts,kf_lfp,linestyle=':',color='r')
  # plt.figure()
  # plt.plot(kf_ts,kf_run.mon['net.Je'],color='r')
  # plt.hlines(net.Je.value,0,ts.max(),color='r',linestyle='--')
  # plt.plot(kf_ts,kf_run.mon['net.Ji'],color='g')
  # plt.hlines(net.Ji.value,0,ts.max(),color='g',linestyle='--')
  # # plt.figure()
  # # plt.plot(input,color='k')
  # # plt.plot(kf_input,linestyle=':',color='r',linewidth=5)
  # plt.show()
