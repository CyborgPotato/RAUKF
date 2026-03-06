import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share

from brainpy.math import cond
import jax.lax as lax
import jax.numpy as jnp
import jax

from itertools import chain

import matplotlib.pyplot as plt

bm.set_platform('cpu')

def sys_input(t):
  t_start = 100
  t_stop = 200
  exp = bm.exp(-(t-t_start)/((t_stop-t_start)/2)) * (t>=t_start)
  return -10.8 * exp #* ((1+bm.sin((t-t_start)*2*np.pi/1000*100)))

class RateNet(bp.DynamicalSystem):
  def __init__(self, n=3, inp_func=None, OU=True):
    super().__init__()
    self.global_input = bm.Variable(1)
    self.pops = bp.dyn.QIF(
      n,
      tau = 1,
      eta = -5.0,
      delta = 0.1,
      J = 15.,
      **({
        # noise parameters
        'x_ou_mean': 1e-0,
        'x_ou_sigma': 2e-1,
        'x_ou_tau': 1.0,
        'y_ou_mean': 1e-0,
        'y_ou_sigma': 2e-1,
        'y_ou_tau': 1.0,
      } if OU else {})
      # state initializer
      # x_initializer = Uniform(max_val=0.05),
      # y_initializer = Uniform(max_val=0.05),      
    )
    self.inp_func = inp_func
    
  def update(self):
    if self.inp_func is None:
      return self.pops(self.global_input)
    else:
      t = bp.share.load('t')
      return self.pops(self.inp_func(t))

import filters.ukf as filt_ukf
from importlib import reload
reload(filt_ukf)
from filters.ukf import Ukf
import re

def dict_get(state, keys):
  for k in keys:
    state = state[k]
  return state

class RAUKF(bp.DynamicalSystem):
  def __init__(self, net, xs=[], ps=[], rs=[], observation=[]):
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
          self.x_map[k]=sk+[k]
    # TODO: refactor below
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
    r0s = []
    for r in rs:
      reg = re.compile(r)
      for x_key in self.x_map.keys():
        if reg.match(x_key):
          r0s.append(x_key)
    self.x0 = sorted(list(set(x0s)))
    self.p0 = sorted(list(set(p0s)))
    self.r0 = sorted(list(set(r0s)))
    self._x = {k:np.array(self.x_vals[k]) for k in self.x0}
    self._p = {k:np.array(self.x_vals[k]) for k in self.p0}
    x = np.concatenate(list(self._x.values())+list(self._p.values()))

    self.t_stab = 0
    self.x = bm.Variable(x.size)
    self.x.value = x
    self.P = bm.Variable((self.x.size,self.x.size))
    self.Q = bm.Variable((self.x.size,self.x.size))
    self.obs = bm.array(observation)
    self.obs_i = bm.Variable(1,dtype=int)
    obs_size = observation.shape[1]
    self.R = bm.Variable((obs_size,obs_size))
    bm.fill_diagonal(self.P,np.array([1e-3]*self.x.size))
    self.P.value = self.P.at[-1,-1].set(1e-1)
    bm.fill_diagonal(self.Q,np.array([1e-2]*self.x.size))
    bm.fill_diagonal(self.R,np.array([1e-2]*obs_size))
    self.phi = bm.Variable(1)
    self.mu = bm.Variable(1)
    self.kappa = 3
    self.resample = True
    self.robust = True
    self.robust_after = 0
    self.threshold = 0.45
    self.lambda0 = 0.2
    self.delta0 = 0.2
    self.a = 5
    self.b = 5

  def unscented_transform(self,x,P):
    N = x.size
    P = (P + P.T) / 2.0
    chol = bm.linalg.cholesky(P)
    chol *= bm.sqrt(N+self.kappa)
    sigmapoints = x.T * bm.ones((2 * N, N)) + bm.concatenate((chol.T, -chol.T))
    if self.kappa != 0:
      sigmapoints = bm.concatenate((x[np.newaxis], sigmapoints))
    return sigmapoints

  def set_x(self,state_dict,x):
    i=0
    for k in chain(self.x0,self.p0):
      ksz = self.x_szs[k]
      v = dict_get(state_dict,self.x_map[k][:-1])
      v[k] = v[k].at[:].set(x[i:i+ksz])
      i+=ksz
    self.net.load_state_dict(state_dict)
    return state_dict
  
  def sample_points(self,state_dict,z_):
    x = bm.concatenate([dict_get(
      state_dict,self.x_map[k]
    ) for k in chain(self.x0,self.p0)])
    z_x = jnp.zeros((z_.shape[0],x.size))
    z_y = jnp.ones((z_.shape[0],self.obs.shape[1]))
    alphas = bm.ones(z_.shape[0])/(2*(x.size+self.kappa))
    alphas = alphas.at[0].set(self.kappa / (x.size+self.kappa))
    for zi,z in enumerate(z_):
      sigmapoint_dict = dict(state_dict)
      self.net.load_state_dict(state_dict)
      self.set_x(sigmapoint_dict,z)
      self.net()
      res_dict = self.net.state_dict()
      sig_x = bm.concatenate([dict_get(
        res_dict,self.x_map[k]
      ) for k in chain(self.x0,self.p0)])
      sig_y = bm.concatenate([dict_get(
        res_dict,self.x_map[k]
      ) for k in self.r0])
      z_x = z_x.at[zi].set(sig_x)
      z_y = z_y.at[zi].set(sig_y)
    self.net.load_state_dict(state_dict)
    z_x = bm.array(z_x)
    z_y = bm.array(z_y)
    return z_x, z_y, alphas
  
  def update(self):
    t_now = bp.share.load('t')
    state_dict = self.net.state_dict()
    z_ = self.unscented_transform(self.x,self.P)
    z_x,z_y,alphas = self.sample_points(state_dict, z_)
    new_x = z_x.T @ alphas
    x_cov = z_x - new_x.T
    new_P = (alphas * x_cov.T)@x_cov + self.Q
    new_P = (new_P+new_P.T)/2
    # Use new_P and new_x?
    if self.resample:
      z_ = self.unscented_transform(new_x,new_P)
      z_x,z_y,alphas = self.sample_points(state_dict, z_)
      new_y = z_y.T @ alphas
    else:
      new_y = z_y.T @ alphas
    # jax.debug.print("{x}",x=z_y)
    y_cov = z_y - new_y.T
    Pyy = (alphas * y_cov.T)@y_cov + self.R
    Pxy = (alphas * x_cov.T)@y_cov
    
    K = Pxy @ bm.linalg.inv(Pyy)
    innovation = self.obs[self.obs_i[0]]-new_y
    # jax.debug.print(
    #   "{y}: {x}",
    #   x=x_cov,
    #   y=y_cov,
    # )

    # jax.debug.print("z_x: {x}",x=z_x)
    # jax.debug.print("alphas: {x}",x=alphas)
    # jax.debug.print("new_x: {x}",x=new_x)
    xhat = new_x + K@innovation
    Phat = new_P - K@Pxy.T

    apply_kf = t_now > self.t_stab
    
    self.x.value = apply_kf*xhat + (1-apply_kf)*self.x
    self.P.value = apply_kf*Phat + (1-apply_kf)*self.P

    if self.robust:
      self.phi.value = self.phi.at[:].set(innovation @ bm.linalg.inv(Pyy) @ innovation.T)
      adapt = bm.any(self.phi > self.threshold)
      adapt *= t_now > self.robust_after
      adapt *= apply_kf
      # Update Q
      phi = self.phi.value[0]
      lambda_ = bm.max(bm.array([
        self.lambda0,
        (phi - self.a * self.threshold) / phi
      ]))
      self.Q.value = ((1 - lambda_) * self.Q + lambda_ * (K @ innovation.T @ (innovation @ K.T)))*adapt + self.Q*(1-adapt)

      # Re-sample sigmapoints with new state estimate
      z_ = self.unscented_transform(self.x,self.P)
      z_x,z_y,alphas = self.sample_points(state_dict, z_)
      new_y = z_y.T @ alphas
      y_cov = z_y - new_y.T
      Pyy = (alphas * y_cov.T)@y_cov + self.R

      # Update R
      # residual_y = y_k - self.plant.observe(x_hat.T)
      # sigma_yy = (self.alphas * Pyy).T @ Pyy
      delta = bm.max(bm.array([
        self.delta0,
        (phi - self.b * self.threshold) / phi
      ]))
      self.R.value = ((1 - delta) * self.R  + delta * (y_cov.T @ y_cov + Pyy))*adapt + self.R*(1-adapt)

      # Correct estimates
      new_x = z_x.T @ alphas
      x_cov = z_x - new_x.T
      new_P = (alphas * x_cov.T)@x_cov + self.Q
      new_P = (new_P+new_P.T)/2
      Pxy = (alphas * x_cov.T)@y_cov
      Pyy = (alphas * y_cov.T)@y_cov + self.R
      K = Pxy @ bm.linalg.inv(Pyy)
      innovation = self.obs[self.obs_i[0]]-new_y

      xhat = new_x + K@innovation
      Phat = new_P - K@Pxy.T

      self.x.value = xhat*adapt + self.x*(1-adapt)
      self.P.value = Phat*adapt + self.P*(1-adapt)

    self.net.load_state_dict(state_dict)
    self.set_x(state_dict,xhat)
    self.net()
    state_dict = self.net.state_dict()
    self.obs_i.value += 1
    return xhat

if __name__=='__main__':
  t_sim = 500
  net = RateNet(1,sys_input,True)
  runner = bp.DSRunner(net, monitors=['pops.y','pops.x'])

  _=runner.run(t_sim)

  observation = np.c_[runner.mon['pops.x']]#,runner.mon['pops.y']]

  net_kf = RAUKF(
    RateNet(1,None,False),
    [ # What internal states to track
      r'QIF\d+\.x',
      r'QIF\d+\.y',
    ],
    [ # What states to estimate
      r'.*global_input$'
    ],
    [ # What our measurement/observation is
      r'QIF\d+\.x',
      # r'QIF\d+\.y',
    ],
    observation
  )
  net_kf.Q.value = np.diag(np.array([
      1e-6,
      1e-10,
      1e-3,
  ],dtype=np.float32))
  net_kf.P.value = np.diag(np.array([
      1e-6,
      1e-10,
      1e-3,
  ],dtype=np.float32))

  net_kf.R.value = np.diag(np.array([
      1e-10,
      # 1e-10,
  ],dtype=np.float32))
  net_kf.robust = False
  net_kf.lambda0 = 0.2
  net_kf.delta0 = 0.2
  net_kf.a = 5
  net_kf.b = 5
  net_kf.threshold = .45

  # net_kf.x = net_kf.x.at[-1].set(3)
  
  runner = bp.DSRunner(net_kf, monitors=['net.pops.x','net.pops.y','P','net.global_input','R','phi'])
  _=runner.run(t_sim)

  plt.plot(observation)
  plt.plot(runner.mon['net.pops.x'])
  # plt.plot(runner.mon['net.pops.y'])
  plt.twinx()
  ts = runner.mon['ts']
  plt.plot(sys_input(ts),color='r')
  plt.plot(runner.mon['net.global_input'],color='k')
  # plt.twinx()
  # plt.plot(runner.mon['R'][:,0,0],color='g')
  # plt.twinx()
  # plt.plot(runner.mon['phi'],color='m')
  plt.show()
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


  # kf = Ukf(
  #   net_kf,
  #   observation,
  #   I_inj,
  #   net_kf.get_pinds(),
  #   net_kf.get_x0(),
  #   net_kf.get_P0(),
  #   net_kf.get_Q0(),
  #   net_kf.get_R0(),
  #   kappa=0,
  #   sigma=0.1,
  #   robust=False,
  #   lambda0=0.1,
  #   delta0=0.1,
  #   a=15,
  #   b=15,
  # )
  # kf.run_estimation()

  # # plt.plot(ts, kf.x[:,0])
  # # until = np.flatnonzero(np.isnan(kf.x[:,0]))
  # # until = None if until.size == 0 else until[0]
  # # plt.plot(ts, observation[:until,0],linestyle='--',color='k')
  # # plt.twinx()
  # # plt.plot(ts, sys_input(ts)*np.ones_like(ts),color='g')
  # # plt.plot(ts, kf.x[:,2],color='r')
  # # plt.show()

  # # plt.plot(kf.P[0::3,0])
  # # plt.plot(kf.P[1::3,1])
  # # plt.plot(kf.P[2::3,2])
  # # plt.show()
