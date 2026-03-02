import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share

from brainpy.math import cond
import jax.lax as lax
import jax.numpy as jnp

import matplotlib.pyplot as plt

bm.set_platform('cpu')

class Exponential(bp.Projection):
  def __init__(self, pre, post, delay, prob, g_max, tau, E):
    super().__init__()
    self.pron = bp.dyn.FullProjAlignPost(
      pre=pre,
      delay=delay,
      # Event-driven computation
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
      syn=bp.dyn.Expon(size=post.num, tau=tau),# Exponential synapse
      out=bp.dyn.COBA(E=E), # COBA network
      post=post
    )

class EINet(bp.DynamicalSystem):
  def __init__(self, ne=3200, ni=800):
    super().__init__()
    lif_pars = dict(V_rest=-60., V_th=-50., V_reset=-70., tau=20.,
                    tau_ref=5.,V_initializer=bp.init.Normal(-55., 4.))
    self.E = bp.dyn.LifRef(ne, **lif_pars)
    self.I = bp.dyn.LifRef(ni, **lif_pars)
    self.E2E = Exponential(self.E, self.E, 0., 0.02, 0.6*2, 2., 0.)
    self.E2I = Exponential(self.E, self.I, 0., 0.02, 0.6*2, 2., 0.)
    self.I2E = Exponential(self.I, self.E, 0., 0.02, 6.7*2, 5., -80.)
    self.I2I = Exponential(self.I, self.I, 0., 0.02, 6.7*2, 5., -80.)
    self.input = bm.Variable(1,batch_axis=0)
    self.input = self.input.at[:].set(1e3)

    self.lfp = bm.Variable(1,batch_axis=0)
    self.calc_lfp()

  def calc_lfp(self):
    self.lfp.value = self.lfp.at[:].set(
      -bm.sum(self.E2E.pron.syn.g) +\
      -bm.sum(self.E2I.pron.syn.g) +\
       bm.sum(self.I2E.pron.syn.g) +\
       bm.sum(self.I2I.pron.syn.g)
    )

  def update(self):
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.calc_lfp()
    self.E(self.input)
    self.I()
    return self.E.spike.value, self.I.spike.value

class DBS(bp.DynamicalSystem):
  def __init__(self, net, tgts, dbs_times, DBS_aff_act, DBS_eff_act):
    super().__init__()
    self.net = net
    self.dbs_times = bm.Variable(np.sort(dbs_times),dtype=float,batch_axis=0)
    self.last_dbs = bm.Variable(1,batch_axis=0)
    self.dbs_idx = bm.Variable(1,dtype=int,batch_axis=0)
    self.tgts = tgts
    self.t_ = bm.Variable(1,batch_axis=0)
    self.DBS_aff_act = bm.Variable(bm.array([DBS_aff_act]))
    self.DBS_eff_act = bm.Variable(bm.array([DBS_eff_act]))
    pres  = {tgt:[] for tgt in tgts}
    posts = {tgt:[] for tgt in tgts}
    for tgt in tgts:
      for v in net.nodes().values():
        try:
          if tgt == v.post:
            comm = v.comm
            src_inds = jnp.repeat(
              jnp.arange(
                comm.indptr.size - 1
              ), jnp.diff(comm.indptr)
            )
            comm.src_inds = src_inds
            comm.dst_inds = comm.indices
            comm.src_uniq = jnp.unique(src_inds)
            comm.dst_uniq = jnp.unique(comm.indices)

            # We will recruit some fraction of synapses surrounding some
            # fraction of the target population. This can be calculated
            # prior to the stimulation starting.
            DBS_aff_act = self.DBS_aff_act.value[0]
            dsts    = comm.dst_uniq
            dst_act = bm.sort(
              bm.random.choice(
                dsts,
                max(int(dsts.size*DBS_aff_act),0)
              )
            )
            dst_mask = jnp.isin(comm.dst_inds,dst_act)
            comm.dst_act_inds = comm.dst_inds.at[dst_mask].get()
            comm.src_act_inds = comm.src_inds.at[dst_mask].get()

            act_indptr = np.zeros_like(comm.indptr)
            sact_inds, sact_counts = np.unique(
              comm.src_act_inds,return_counts=True
            )

            act_indptr[sact_inds+1] = sact_counts
            act_indptr = np.cumsum(act_indptr)
            
            comm.act_indices = comm.dst_act_inds
            comm.act_indptr = act_indptr
            ## Each Pulse some fraction of synapses will be
            ## activated. This can either be calculated for each DBS
            ## pulse or for all pulses separately prior to the
            ## simulation starting This will require precalculating
            ## the array size, e.g. if there are 1000 afferent
            ## synapses, and we want to say some fixed probability of
            ## recruitment of a synapse for each DBS pulse then we
            ## need to store that value, e.g. 80% -> 800, prior to
            ## running the simulation.
            ### DBS_aff_syn_act = 0.8
            ### dst_mask *=
            ### bm.random.bernoulli(DBS_aff_syn_act,size=dst_inds)
            posts[tgt].append(v)
        except Exception as e:
          if not isinstance(e,AttributeError):
            print(e)
        try:
          if tgt == v.pre:
            comm = v.comm
            src_inds = jnp.repeat(
              jnp.arange(
                comm.indptr.size - 1
              ), jnp.diff(comm.indptr)
            )
            comm.src_inds = src_inds
            comm.dst_inds = comm.indices
            comm.src_uniq = jnp.unique(src_inds)
            comm.dst_uniq = jnp.unique(comm.indices)

            # We will recruit some fraction of synapses surrounding some
            # fraction of the target population. This can be calculated
            # prior to the stimulation starting.
            DBS_eff_act = self.DBS_eff_act.value[0]
            srcs    = comm.src_uniq
            src_act = bm.sort(
              bm.random.choice(
                srcs,
                max(int(srcs.size*DBS_eff_act),0)
              )
            )
            src_mask = jnp.isin(comm.src_inds,src_act)
            comm.dst_act_inds = comm.dst_inds.at[src_mask].get()
            comm.src_act_inds = comm.src_inds.at[src_mask].get()

            act_indptr = np.zeros_like(comm.indptr)
            sact_inds, sact_counts = np.unique(
              comm.src_act_inds,return_counts=True
            )

            act_indptr[sact_inds+1] = sact_counts
            act_indptr = np.cumsum(act_indptr)
            
            comm.act_indices = comm.dst_act_inds
            comm.act_indptr = act_indptr
            pres[tgt].append(v)
        except Exception as e:
          if not isinstance(e,AttributeError):
            print(e)
    self.pres  = pres
    self.posts = posts

  def update(self):
    self.t_.value = self.t_ + share.dt
    n_past = self.dbs_times <= self.t_
    n_pre  = self.dbs_times > self.last_dbs
    n_dbs  = (n_past & n_pre).sum()
    self.dbs_idx.value += n_dbs
    def nop(*args):
      return
    def upd_dbs():
      self.last_dbs.value = jnp.take(self.dbs_times,self.dbs_idx)
    cond(n_dbs>0,upd_dbs,nop)
    def _pulse(connection):
      comm = connection.comm
      syn = connection.syn
      def pulse(x):
        old_indices = comm.indices
        old_indptr = comm.indptr
        comm.indices = comm.act_indices
        comm.indptr = comm.act_indptr
        delta = comm.update(x)
        comm.indices = old_indices
        comm.indptr = old_indptr
        syn.add_current(delta)
      return pulse

    for tgt in self.tgts:
      # Where the target is post/destination so DBS will activate some
      # fraction of the afferent indices
      for post in self.posts[tgt]:
        x = np.ones(post.pre.size)*(n_dbs>0)
        pulse = _pulse(post)
        # May be more GPU performant to not perform the conditional
        # no-op: alternatively use x where
        # x=np.ones(post.pre.size)*(n_dbs>0)
        pulse(x)
        # cond(n_dbs>0,pulse,nop,x)
      for pre in self.pres[tgt]:
        x = np.ones(pre.pre.size)*(n_dbs>0)
        pulse = _pulse(pre)
        pulse(x)
    return self.net()

import main as main_
from importlib import reload
reload(main_)
from main import RAUKF, Ukf

net = EINet()

net_dbs = DBS(net,[net.E],np.arange(2500,2600,100),0.1,0)

runner = bp.DSRunner(net_dbs, monitors=['net.lfp'])

_=runner.run(.5e3)

observation = runner.mon['net.lfp']
I_inj = 0*observation

net_kf = RAUKF(
  DBS(EINet(),[net.E],np.arange(2500,2600,100),0.5,0),
  [ # What internal states to track
    r'.*lfp$',
    # r'.*V$',
  ],
  [ # What states to estimate
    # r'.*input$',
    r'.*DBS_aff_act$',
  ],
  [
    r'.*lfp$',
  ],
  observation,
)
net_kf.robust = False
net_kf.x.value = net_kf.x.at[-1].set(0.5)
# net_kf.P.value = net_kf.P.at[0,0].set(1e1)
# net_kf.P.value = net_kf.P.at[-1,-1].set(1e-3)
# net_kf.Q.value = net_kf.Q.at[0,0].set(1e-3)
# net_kf.Q.value = net_kf.Q.at[-1,-1].set(1e-3)
net_kf.R.value = net_kf.R.at[-1,-1].set(1e-6)
kf_run = bp.DSRunner(net_kf, monitors=['net.net.lfp','net.DBS_aff_act'])
_=kf_run.run(.5e3)
plt.plot(runner.mon['net.lfp'],color='g')
plt.plot(kf_run.mon['net.net.lfp'],color='k')
plt.twinx()
plt.plot(kf_run.mon['net.DBS_aff_act'],color='r')
# plt.figure()
# bp.visualize.raster_plot(runner.mon.ts,runner.mon['net.I.spike'])
# plt.xlim(2490,2510)
# plt.figure()
# bp.visualize.raster_plot(runner.mon.ts,runner.mon['net.E.spike'])
# plt.xlim(2490,2510)
plt.show()
