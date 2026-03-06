import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
import brainstate.random as bsr

from brainpy.math import cond
import jax.lax as lax
import jax.numpy as jnp

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
