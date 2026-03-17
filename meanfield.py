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
    dwdt = (self.a * 1e-6 *  (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def dV(self, V, t, w, I):
    exp = self.gL * 1e-6 * self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dVdt = (-self.gL * 1e-6 * (V - self.V_rest) + exp - w + I) / (self.tau*1e-9)
    return dVdt

  def update(self, x=None):
    t = share.load('t')/1e3
    dt = share.load('dt')/1e3
    x = 0. if x is None else x
    x = self.sum_current_inputs(self.V.value, init=x)

    # integrate membrane potential
    V, w = self.integral(self.V.value, self.w.value, t, x, dt)
    V += self.sum_delta_inputs()

    # spike, spiking time, and membrane potential reset
    if isinstance(self.mode, bm.TrainingMode):
      spike = self.spk_fun(V - self.V_th)
      spike = stop_gradient(spike) if self.detach_spk else spike
      if self.spk_reset == 'soft':
        V -= (self.V_th - self.V_reset) * spike
      elif self.spk_reset == 'hard':
        V += (self.V_reset - V) * spike
      else:
        raise ValueError(f"Unknown spk_reset mode: {self.spk_reset}. Must be 'soft' or 'hard'.")
      w += self.b * spike
    else:
      spike = V >= self.V_th
      V = bm.where(spike, self.V_reset, V)
      w = bm.where(spike, w + self.b, w)

    self.V.value = V
    self.w.value = w
    self.spike.value = spike
    return spike    

class SNN(bp.DynamicalSystem):
  def __init__(self, noise=True, ne=8000, ni=2000):
    super().__init__()
    self.lfp = bm.Variable(1)
    cmn_args = {
      "V_rest": -63,
      "V_reset": -65,
      "V_T": -50,
      "V_th": -50,
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

    self.E2E = Exponential(self.E, self.E, 2, 0.05, 1.0e-6, 5e-3,   0.)
    self.E2I = Exponential(self.E, self.I, 2, 0.05, 1.0e-6, 5e-3,   0.)
    self.I2E = Exponential(self.I, self.E, 2, 0.05, 5.0e-6, 5e-3, -80.)
    self.I2I = Exponential(self.I, self.I, 2, 0.05, 5.0e-6, 5e-3, -80.)

    self.NE = bp.dyn.PoissonInput(self.E2E.pron.syn.g,400,1e-6,5e-3)
    self.NI = bp.dyn.PoissonInput(self.E2I.pron.syn.g,400,1e-6,5e-3)
    
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
    
    self.ve  = bm.Variable(bm.array([0.]))
    self.vi  = bm.Variable(bm.array([0.]))
    self.we  = bm.Variable(1)
    self.wi  = bm.Variable(1)

    self.Pe = bm.Variable(bm.array([
      -49.8, 5.06, -25, 1.4, -0.41, 10.5, -36, 7.4, 1.2, -40.7
    ]))*1e-3
    self.Pi = bm.Variable(bm.array([
      -51.4, 4.0, -8.3, 0.2, -0.5, 1.4, -14.6, 4.5, 2.8, -15.3
    ]))*1e-3
    # TODO Set these variables
    self.g = bm.Variable(bm.array([0.2]))
    self.pe = bm.Variable(bm.array([0.05]))
    self.pi = bm.Variable(bm.array([0.05]))
    self.N_tot = bm.Variable(bm.array([10000]))
    self.N_ext_e = bm.Variable(bm.array([500]))
    self.N_ext_i = bm.Variable(bm.array([500]))
    self.Fe_ext = bm.Variable(bm.array([1.]))
    self.Fi_ext = bm.Variable(bm.array([1.]))
    self.tau_e = bm.Variable(bm.array([5e-3]))
    self.Qe = bm.Variable(bm.array([1.5e-9]))
    self.tau_i = bm.Variable(bm.array([5e-3]))
    self.Qi = bm.Variable(bm.array([5e-9]))
    self.gL = bm.Variable(bm.array([10e-9]))
    self.Ee = bm.Variable(bm.array([0.]))
    self.Ei = bm.Variable(bm.array([-80e-3]))
    self.ELe = bm.Variable(bm.array([-63e-3]))
    self.ELi = bm.Variable(bm.array([-63e-3]))
    self.Cm = bm.Variable(bm.array([281e-9]))
    #
    self.uv0  = bm.Variable(bm.array([-60e-3]))
    self.stdv0 = bm.Variable(bm.array([0.004e-3]))
    self.tauNv0 = bm.Variable(bm.array([0.5]))
    self.duv  = bm.Variable(bm.array([0.001e-3]))
    self.dstdv = bm.Variable(bm.array([0.006e-3]))
    self.dtauNv = bm.Variable(bm.array([1.]))
    self.T = bm.Variable(bm.array([50e-3]))
    self.tau_we = bm.Variable(bm.array([500e-3]))
    self.tau_wi = bm.Variable(bm.array([500e-3]))
    self.be = bm.Variable(bm.array([30e-12]))
    self.ae = bm.Variable(bm.array([4e-12]))
    self.bi = bm.Variable(bm.array([0.]))
    self.ai = bm.Variable(bm.array([0.]))

    self.int_v = bp.odeint(self.dv)
    self.int_w = bp.odeint(self.dw)

  def dv(self, v, t, uv, stdv, tauv, P):
    return (self.F(v,uv,stdv,tauv,P)-v)/self.T

  def dw(self, w, t, v, a, b, tau_w, uv, EL):
    return -w/tau_w + b*v + a*(uv-EL)/tau_w

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

  def F(self, v, uv, stdv, tauv, P):
    return jax.lax.erfc(
      (1e3*self.Veff_th(uv,stdv,tauv,P)-uv) / (bm.sqrt(2)*stdv)
    ) / (2*tauv)

  def volt_dyn(self, ve, vi, w):
    g=self.g
    pe=self.pe
    pi=self.pi
    N_tot=self.N_tot
    Fe_ext=self.Fe_ext
    N_ext_e=self.N_ext_e
    Fi_ext=self.Fi_ext
    N_ext_i=self.N_ext_i
    Qe=self.Qe
    Qi=self.Qi
    tau_e=self.tau_e
    tau_i=self.tau_i
    gL=self.gL
    Cm=self.Cm
    Ee=self.Ee
    Ei=self.Ei
    EL=self.ELe
    # firing rate
    # 1e-6 represent spontaneous release of synaptic neurotransmitter
    # or some intrinsic currents of neurons
    fe = (ve + 1.0e-6) * (1. - g) * pe * N_tot + Fe_ext * N_ext_e
    fi = (vi + 1.0e-6) * g * pi * N_tot + Fi_ext * N_ext_i

    # conductance fluctuation and effective membrane time constant
    mu_Ge, mu_Gi = Qe * tau_e * fe, Qi * tau_i * fi
    mu_G = gL + mu_Ge + mu_Gi
    T_m = Cm / mu_G

    # membrane potential
    mu_V = (mu_Ge * Ee + mu_Gi * Ei + gL * EL - w) / mu_G
    # post-synaptic membrane potential event s around muV
    U_e, U_i = Qe / mu_G * (Ee - mu_V), Qi / mu_G * (Ei - mu_V)
    # Standard deviation of the fluctuations
    sigma_V = bm.sqrt(
        fe * (U_e * tau_e) ** 2 / (2. * (tau_e + T_m)) + fi * (U_i * tau_i) ** 2 / (2. * (tau_i + T_m)))
    # Autocorrelation-time of the fluctuations
    T_V_numerator = (fe * (U_e * tau_e) ** 2 + fi * (U_i * tau_i) ** 2)
    T_V_denominator = (fe * (U_e * tau_e) ** 2 / (tau_e + T_m) + fi * (U_i * tau_i) ** 2 / (tau_i + T_m))
    T_V = T_V_numerator / T_V_denominator
    return mu_V, sigma_V, T_V

  def update(self, x=None):
    t = bp.share['t']/1000
    dt = bp.share['dt']/1000

    volt_dyn_e = self.volt_dyn(self.ve,self.vi,self.we)
    volt_dyn_i = self.volt_dyn(self.ve,self.vi,self.we)

    ve = self.int_v(self.ve,t,*volt_dyn_e,self.Pe,dt=dt)
    vi = self.int_v(self.vi,t,*volt_dyn_i,self.Pi,dt=dt)
    we = self.int_w(self.we,t,self.ve,self.ae,self.be,self.tau_we,volt_dyn_e[0],self.ELe,dt=dt)
    wi = self.int_w(self.wi,t,self.vi,self.ai,self.bi,self.tau_wi,volt_dyn_i[0],self.ELi,dt=dt)

    self.ve.value = ve
    self.vi.value = vi
    self.we.value = we
    self.wi.value = wi
  
if __name__=='__main__':
  t_stop = 2e3

  # mean_net = MeanField()
  # mean_run = bp.DSRunner(mean_net, monitors=['ve','vi'])

  # _=mean_run.run(t_stop)
  # ve = mean_run.mon['ve']
  # vi = mean_run.mon['vi']
  # plt.plot(ve)
  # plt.plot(vi)
  # plt.show()
  net = SNN()

  snn_run = bp.DSRunner(net, monitors=['lfp','E.spike','I.spike'])

  _=snn_run.run(t_stop)

  observation = snn_run.mon['lfp']

  obs = observation#+np.random.normal(0,50,size=observation.shape)

  plt.plot(obs)
  plt.scatter(*snn_run.mon['E.spike'].nonzero(),marker='|')
  plt.scatter(*snn_run.mon['I.spike'].nonzero(),marker='|')
  plt.show()
