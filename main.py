import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share

from brainpy.math import cond
import jax.lax as lax
import jax.numpy as jnp

from itertools import chain

import matplotlib.pyplot as plt

bm.set_platform('cpu')

def sys_input(t):
  t_start = 100
  t_stop = 200
  exp = bm.exp(-(t-t_start)/((t_stop-t_start)/2)) * (t>=t_start)
  return -2.8 * exp #* ((1+bm.sin((t-t_start)*2*np.pi/1000*100)))

class RateNet(bp.DynamicalSystem):
  def __init__(self, n=3):
    super().__init__()
    self.pops = bp.dyn.QIF(
      n,
      tau = 1,
      eta = -5.0,
      delta = 0.1,
      J = 15.,
      # noise parameters
      x_ou_mean = 1e-0,
      x_ou_sigma = 2e-1,
      x_ou_tau = 1.0,
      y_ou_mean = 1e-0,
      y_ou_sigma = 2e-1,
      y_ou_tau = 1.0,

      # state initializer
      # x_initializer = Uniform(max_val=0.05),
      # y_initializer = Uniform(max_val=0.05),      
    )
    
  def update(self, inp=None):
    if inp is None:
      t = bp.share.load('t')
      return self.pops(sys_input(t)
        #((1+bm.sin(t*2*np.pi/1000*50)))*-1.8 * (
        #  1#(t>100)*(t<200)
        #)
      )
    else:
      return self.pops(inp)

import filters.ukf as filt_ukf
from importlib import reload
reload(filt_ukf)
from filters.ukf import Ukf
import re
  
class RAUKF(bp.DynamicalSystem):
  def __init__(self, net, xs=[], ps=[]):
    super().__init__()
    self.net = net
    self.net_state = self.net.state_dict()
    self.x_map = {}
    self.x_vals = {}
    self.x_szs = {}
    stack = [(self.net_state,[])]
    while len(stack) != 0:
      s,sk = stack.pop()
      for k,v in s.items():
        if isinstance(v,dict):
          stack.append((v,sk+[k]))
        else:
          self.x_szs[k]=v.size
          self.x_vals[k]=v
          self.x_map[k]=sk
    x0s = []
    for x in xs:
      reg = re.compile(x)
      for x_key in self.x_map.keys():
        if reg.match(x_key):
          x0s.append(x_key)
    p0s = []
    for p in ps:
      reg = re.compile(p)
      for x_key in self.x_map.keys():
        if reg.match(x_key):
          p0s.append(x_key)
    self.x0 = sorted(list(set(x0s)))
    self.p0 = sorted(list(set(p0s)))
    self.p = {k:np.array(self.x_vals[k]) for k in self.p0}
    self._forward_jit = self._forward#bm.jit(self._forward)

  def get_p0(self):
    return np.concatenate([np.array(self.x_vals[x]) for x in self.p0])

  def get_x0(self, nop0=False):
    x0 = np.concatenate([np.array(self.x_vals[x]) for x in self.x0])
    if nop0:
      return x0[np.newaxis]
    return np.concatenate([x0,self.get_p0()])[np.newaxis]

  def get_pinds(self):
    return np.concatenate(list(self.p.values()),dtype=int)

  def get_x0_size(self, nop0=False):
    ret = sum([self.x_vals[k].size for k in net_kf.x0])
    if nop0:
      return ret
    return ret+sum([self.x_vals[k].size for k in net_kf.p0])

  def get_P0(self):
    return np.diag(np.ones(self.get_x0_size()))
  def get_Q0(self):
    return np.diag(np.ones(self.get_x0_size()))
  def get_R0(self):
    # Observation noise
    return np.diag(np.ones(2)*1e-1)
    return np.diag(np.ones(1)*1e-2)

  def _forward(self):
    self.net(1)

  def forward(self, xks, Ik, p, int_factor=1):
    x_keys = self.x0
    net_state = self.net_state
    ret = np.zeros_like(xks)
    
    new_state = dict(net_state)
    breakpoint()
    for (i,x1),(k,p1) in zip(enumerate(xks),enumerate(p)):
      for k,v in zip(x_keys,x1):
        kv = new_state
        for kj in self.x_map[k]:
          kv = kv[kj]
        kv[k] = kv[k].at[:].set(v)
      for k,v in zip(self.p0,p1):
        kv = new_state
        for kj in self.x_map[k]:
          kv = kv[kj]
        kv[k] = kv[k].at[:].set(v)
      # self.net.load_state_dict(new_state)
      # self._forward_jit()
      upd_state = self.net.state_dict()
      for j,k in enumerate(x_keys):
        kv = upd_state
        for kj in self.x_map[k]:
          kv = kv[kj]
        ret[i,j] = np.array(kv[k])[0]
    return ret

  def observe(self, x):
    return x[:,:2]

if __name__=='__main__':
  net = RateNet(2)

  net_kf = RAUKF(
    RateNet(2),
    [ # What internal states to track
      r'QIF\d+\.x',
      r'QIF\d+\.y',
    ],
    [ # What states to estimate
      r'QIF\d+\.input$'
    ],
  )

  runner = bp.DSRunner(net, monitors=['pops.y','pops.x'])

  _=runner.run(300)

  observation = runner.mon['pops.x']#,runner.mon['pops.y']]
  I_inj = np.zeros_like(observation)

  ts = np.arange(observation.shape[0])*bp.share.load('dt')

  # plt.figure()
  # plt.xlabel('Time (ms)')
  # plt.ylabel('Voltage (mV)')
  # plt.plot(ts, observation, color='k')
  # plt.vlines(
  #   100,
  #   observation.min(),
  #   observation.max(),
  #   color='r',
  #   linewidth=2,
  # )
  # plt.show()


  kf = Ukf(
    net_kf,
    observation,
    I_inj,
    net_kf.get_pinds(),
    net_kf.get_x0(),
    net_kf.get_P0(),
    net_kf.get_Q0(),
    net_kf.get_R0(),
    kappa=0,
    sigma=0.1,
    robust=False,
    lambda0=0.1,
    delta0=0.1,
    a=15,
    b=15,
  )
  kf.run_estimation()

  # plt.plot(ts, kf.x[:,0])
  # until = np.flatnonzero(np.isnan(kf.x[:,0]))
  # until = None if until.size == 0 else until[0]
  # plt.plot(ts, observation[:until,0],linestyle='--',color='k')
  # plt.twinx()
  # plt.plot(ts, sys_input(ts)*np.ones_like(ts),color='g')
  # plt.plot(ts, kf.x[:,2],color='r')
  # plt.show()

  # plt.plot(kf.P[0::3,0])
  # plt.plot(kf.P[1::3,1])
  # plt.plot(kf.P[2::3,2])
  # plt.show()
