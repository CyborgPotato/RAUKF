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
    # self.NE = lambda: 100/3
    # self.NI = lambda: 100/3
    
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

# import main as main_
# from importlib import reload
# reload(main_)
from main import RAUKF, Ukf
# import dbs as dbs_
# reload(dbs_)
from dbs import DBS
import gc
# gc.set_debug(gc.DEBUG_LEAK)

def fit_lfp_obs(
    obs=0,t_stop=0,b=0.8,c=0.05,T=1,t_stab=0,progress=False,
    Je_scale=1,Ji_scale=1,dbs_times=np.zeros(1),
    dbs_tgts='E', dbs_pct_aff=0.1, dbs_pct_eff=0.1,
    justEI=True, R_obs=["lfp_max","lfp_min"],index=0,
    aff_scale=1,eff_scale=1,
):
  if justEI:
    states = [
      r'.*Je$',
      r'.*Ji$',
    ]
  else:
    states = [
      r'.*Je$',
      r'.*Jee$',
      r'.*Ji$',
      r'.*Jii$',
    ]

  states += [
    r'.*DBS_aff_act',
    r'.*DBS_eff_act',
  ]
    
  net_kf_ = EINet(b=b,c=c)

  if dbs_tgts=='E':
    dbs_tgt=[net_kf_.E]
  elif dbs_tgts=='I':
    dbs_tgt=[net_kf_.I]
  elif dbs_tgts=='EI':
    dbs_tgt=[net_kf_.E,net_kf_.I]

  net_kf = RAUKF(
    # net_kf_,
    DBS(net_kf_,dbs_tgt,dbs_times,dbs_pct_aff,dbs_pct_eff),
    [ # What internal states to track
      # r'.*lfp$',
    ],
    # What states to estimate
    states
    ,
    [ # What our measurement/observation is
      fr'.*{R}$' for R in R_obs
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

  we_Q = 1e-14*2*2*2*2*2*2*2*2*2*2#.25e-10 + 0.125e-10
  wi_Q = 1e-14*2*2*2*2*2*2*2*2*2*2#.25e-10 + 0.125e-10
  aff_Q = 1e-11
  eff_Q = aff_Q
  if justEI:
    net_kf.x.value = net_kf.x.at[-4].set(net_kf.x.value[-4]*aff_scale)
    net_kf.x.value = net_kf.x.at[-3].set(net_kf.x.value[-3]*eff_scale)
    net_kf.x.value = net_kf.x.at[-2].set(net_kf.x.value[-2]*Je_scale)
    net_kf.x.value = net_kf.x.at[-1].set(net_kf.x.value[-1]*Ji_scale)
  else:
    net_kf.x.value = net_kf.x.at[-6].set(net_kf.x.value[-6]*aff_scale)
    net_kf.x.value = net_kf.x.at[-5].set(net_kf.x.value[-5]*eff_scale)
    net_kf.x.value = net_kf.x.at[-4].set(net_kf.x.value[-4]*Je_scale)
    net_kf.x.value = net_kf.x.at[-3].set(net_kf.x.value[-3]*Je_scale)
    net_kf.x.value = net_kf.x.at[-2].set(net_kf.x.value[-2]*Ji_scale)
    net_kf.x.value = net_kf.x.at[-1].set(net_kf.x.value[-1]*Ji_scale)    
  if justEI:
    net_kf.Q.value = np.diag(np.array([
      aff_Q,
      eff_Q,
      we_Q,
      wi_Q,
    ],dtype=np.float32))
    net_kf.P.value = np.diag(np.array([
      aff_Q,
      eff_Q,
      we_Q,
      wi_Q,
    ],dtype=np.float32))
  else:
    e_s = 2
    i_s = 1
    ee_s = 1/2
    ii_s = 2*1.5
    net_kf.Q.value = np.diag(np.array([
      aff_Q,
      eff_Q,
      we_Q*e_s,
      we_Q*ee_s,
      wi_Q*i_s,
      wi_Q*ii_s,
    ],dtype=np.float32))
    net_kf.P.value = np.diag(np.array([
      aff_Q,
      eff_Q,
      we_Q*e_s,
      we_Q*ee_s,
      wi_Q*i_s,
      wi_Q*ii_s,
    ],dtype=np.float32))

  for i in range(len(R_obs)):
    net_kf.R.value = net_kf.R.at[i,i].set(0.01)

  kf_run = bp.DSRunner(
    net_kf, monitors=[
      'net.net.lfp',
      'net.net.lfp_max',
      'net.net.lfp_min',
      'net.net.Jee',
      'net.net.Je',
      'net.net.Jii',
      'net.net.Ji',
      'net.DBS_aff_act',
      'net.DBS_eff_act',
      'P',
      'phi',
    ],
    progress_bar=progress,
  )

  _=kf_run.run(t_stop/net_kf.T)
  return kf_run

def rmse(a,tgt):
  return np.sqrt(np.mean(np.square(a-tgt)))

import inspect
from functools import lru_cache
from hashlib import sha512

class HashDict(dict):
  def __hash__(self):
    return hash(",".join(f"{k}:{v}" for k,v in self.items()))
class HashArr(np.ndarray):
  def __new__(self,arr):
    return np.ndarray(arr.shape,buffer=arr,dtype=arr.dtype).view(HashArr)
  def __hash__(self):
    return hash(sha512(bytes(self)).digest())
class HashList(list):
  def __hash__(self):
    return hash(",".join(f'{v}' for v in self))

default_kf_args = inspect.getfullargspec(fit_lfp_obs)
default_kf_args = {
  k:v for k,v in zip(default_kf_args.args,default_kf_args.defaults)
}
default_ei_args = inspect.getfullargspec(EINet)
default_ei_args = {
  k:v for k,v in zip(default_ei_args.args[1:],default_ei_args.defaults)
}
default_ei_args['index']=0
default_ei_args['progress']=False

@memory.cache
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
    net.net.Jii.value,
    net.DBS_aff_act.value,
    net.DBS_eff_act.value,
  )

def run(args):
  return _run(args)

@memory.cache
def _run(args):
  kf_args, ei_args = args
  for k in default_kf_args.keys():
    if not k in kf_args:
      kf_args[k] = default_kf_args[k]
  for k in default_ei_args.keys():
    if not k in ei_args:
      ei_args[k] = default_ei_args[k]

  t_stop      = kf_args['t_stop']
  dbs_times   = kf_args['dbs_times']
  dbs_tgts    = kf_args['dbs_tgts']
  dbs_pct_aff = kf_args['dbs_pct_aff']
  dbs_pct_eff = kf_args['dbs_pct_eff']
  R_obs       = kf_args['R_obs']

  index       = ei_args.pop('index')
  progress       = ei_args.pop('progress')
      
  ts,obs,(rJe,rJee,rJi,rJii,aff,eff) = generate_obs(
    ei_args,
    t_stop,
    dbs_times,
    dbs_tgts,dbs_pct_aff,dbs_pct_eff,
    R_obs,
    index,
    progress,
  )

  index_dict = dict(kf_args)
  index_dict.update({f'sim_{k}':v for k,v in ei_args.items()})

  kf_args['obs'] = obs
  kf_run = fit_lfp_obs(**kf_args)

  index_dict['RMSE_Je'] = rmse(kf_run.mon['net.net.Je'],rJe)
  index_dict['RMSE_Jee'] = rmse(kf_run.mon['net.net.Jee'],rJee)
  index_dict['RMSE_Ji'] = rmse(kf_run.mon['net.net.Ji'],rJi)
  index_dict['RMSE_Jii'] = rmse(kf_run.mon['net.net.Jii'],rJii)

  # return obs,kf_run

  # breakpoint()

  return pd.DataFrame.from_dict(index_dict,'index').T

t_stop=10000
jes=2
jis=0.1
affs=0.1
effs=0.1
# obs1,nostim_max_min = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.zeros(1),'progress':True,'R_obs':['lfp_max','lfp_min'],'aff_scale':affs,'eff_scale':effs,'dbs_pct_aff':0.5,'dbs_pct_eff':0.5},{'progress':True}))
# obs2,nostim_lfp = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.zeros(1),'progress':True,'R_obs':['lfp']},{'progress':True}))
# obs3,stim_max_min = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.arange(2500,7500,100),'progress':True,'R_obs':['lfp_max','lfp_min'],'aff_scale':affs,'eff_scale':effs,'dbs_pct_aff':0.,'dbs_pct_eff':0.05},{'progress':True}))
obs3,stim_max_min = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.arange(2500,7500,100),'progress':True,'R_obs':['lfp_max','lfp_min'],'aff_scale':affs,'eff_scale':effs,'dbs_pct_aff':0.05,'dbs_pct_eff':0.05,'justEI':False},{'progress':True}))
# plt.plot(stim_max_min.mon['net.DBS_aff_act'])

plt.plot(np.abs(stim_max_min.mon['net.DBS_eff_act']),color='k')
# plt.plot(np.abs(stim_max_min.mon['net.DBS_aff_act']),color='k',linestyle='--')
plt.hlines(0.05,0,t_stop*10,color='r')
plt.fill_betweenx([0,0.1],[25000,25000],[75000,75000],color='k',alpha=0.5)
plt.ylabel('Estimated % Recruited by DBS')
plt.xlabel('Time Step (#)')
plt.figure()
plt.plot(stim_max_min.mon['net.net.Ji'])
plt.plot(stim_max_min.mon['net.net.Jii'],linestyle='--')
plt.plot(stim_max_min.mon['net.net.Je'])
plt.plot(stim_max_min.mon['net.net.Jee'],linestyle='--')
plt.figure()
plt.plot(obs3[:,0],color='r')
plt.plot(stim_max_min.mon['net.net.lfp_max'],color='k')
plt.show()
# obs4,stim_lfp = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.arange(2500,7500,100),'progress':True,'R_obs':['lfp']},{'progress':True}))

# # obs5,stim_max_min_lfp = run(({'t_stop':t_stop,'Je_scale':jes,'Ji_scale':jis,'dbs_times':np.arange(2500,7500,100),'progress':True,'R_obs':['lfp_max','lfp_min','lfp']},{'progress':True}))

# plt.plot(nostim_max_min.mon['net.net.Je'],color='k')
# plt.plot(nostim_lfp.mon['net.net.Je'],color='k',linestyle='--')
# plt.plot(stim_max_min.mon['net.net.Je'],color='r')
# plt.plot(stim_lfp.mon['net.net.Je'],color='r',linestyle='--')
# plt.hlines(1,0,t_stop*10,color='b',linestyle='-.')
# plt.figure()
# plt.plot(nostim_max_min.mon['net.net.Ji'],color='k')
# plt.plot(nostim_lfp.mon['net.net.Ji'],color='k',linestyle='--')
# plt.plot(stim_max_min.mon['net.net.Ji'],color='r')
# plt.plot(stim_lfp.mon['net.net.Ji'],color='r',linestyle='--')
# plt.hlines(-6,0,t_stop*10,color='b',linestyle='-.')
# plt.show()

# from itertools import product, tee, chain

# def dictProduct(**kwargs):
#   ks = kwargs.keys()
#   for vs in product(*kwargs.values()):
#     yield dict(zip(ks,vs))

if __name__=='__main__':
  def produce_params(b):
    kf_arg_ranges = {
      't_stop': [10000],
      'b': [b],
      # 'Je_scale': np.logspace(-1,1,5),
      # 'Ji_scale': np.logspace(-1,1,5),
      'Je_scale': [2],
      'Ji_scale': [2],
      'justEI': [True],
      'index': range(50),
    }
    kf_arg_ranges_dbs = dict(kf_arg_ranges)
    kf_arg_ranges_dbs.update({
      'dbs_times': [
        np.arange(2500,7500,100),
      ],    
      'dbs_tgts': ['EI'],
      'dbs_pct_aff': [0.1],#,0.1,0.2],
      'dbs_pct_eff': [0.1],#,,0.1,0.2],
    })
    kf_arg_ranges_dbs2 = []
    for _ in range(5):
      dbs = dict(kf_arg_ranges)
      dbs.update({
        'dbs_times': [
          np.arange(2500,7500,100)+np.random.uniform(-25,25,size=50),
        ],    
        'dbs_tgts': ['EI'],
        'dbs_pct_aff': [0.1],#,0.1,0.2],
        'dbs_pct_eff': [0.1],#,,0.1,0.2],
        'index': range(10),
      })
      kf_arg_ranges_dbs2.append(dictProduct(**dbs))

    ei_arg_ranges = {
      'b': [b],
      'index': range(1),
    }

    kf_args = chain(
      dictProduct(**kf_arg_ranges),
      dictProduct(**kf_arg_ranges_dbs),
      *kf_arg_ranges_dbs2,
    )

#     ei_args = dictProduct(**ei_arg_ranges)
#     args = product(kf_args,ei_args)
#     return args
#   args = chain(*[produce_params(b) for b in [0.8]])
#   args, _args = tee(args)
#   n_sim = sum(1 for _ in _args)

#   ctx = get_context('forkserver')
#   with ctx.Pool(16,maxtasksperchild=5) as p:
#     runs = p.imap(run,args)
#     results = pd.concat(tqdm(runs,total=n_sim),ignore_index=True)

#   results.to_csv('./raukf_sweep.csv')
    
# #   # kf_lfp = kf_run.mon['net.lfp']
# #   # # kf_input = kf_run.mon['net.Einput']
# #   # kf_ts = kf_run.mon['ts']*kf_T
# #   # stab_mask = kf_ts>=t_stab

# #   # plt.plot(ts,observation,color='k')
# #   # plt.plot(ts,obs,color='g')
# #   # plt.plot(kf_ts,kf_lfp,linestyle=':',color='r')
# #   # plt.figure()
# #   # plt.plot(kf_ts,kf_run.mon['net.Je'],color='r')
# #   # plt.hlines(net.Je.value,0,ts.max(),color='r',linestyle='--')
# #   # plt.plot(kf_ts,kf_run.mon['net.Ji'],color='g')
# #   # plt.hlines(net.Ji.value,0,ts.max(),color='g',linestyle='--')
# #   # # plt.figure()
# #   # # plt.plot(input,color='k')
# #   # # plt.plot(kf_input,linestyle=':',color='r',linewidth=5)
# #   # plt.show()
