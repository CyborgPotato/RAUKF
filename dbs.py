import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
import brainstate.random as bsr

from brainpy.math import cond
import jax
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
    self._DBS_iter = bm.Variable(1,batch_axis=0)
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
            comm.sorted_src_inds = jnp.sort(comm.src_inds)
            comm.dst_inds = comm.indices
            comm.sorted_dst_inds = jnp.sort(comm.dst_inds)
            comm.src_uniq = jnp.unique(src_inds)
            comm.dst_uniq = jnp.unique(comm.indices)
            if   isinstance(DBS_aff_act, list):
              for pop,pct in DBS_aff_act:
                if pop == v.pre:
                  comm.DBS_aff_act = bm.Variable(bm.array([pct]))
                  break
            elif isinstance(DBS_aff_act,float):
              comm.DBS_aff_act = bm.Variable(bm.array([DBS_aff_act]))
            if   isinstance(DBS_eff_act, list):
              for pop,pct in DBS_eff_act:
                if pop == v.pre:
                  comm.DBS_eff_act = bm.Variable(bm.array([pct]))
                  break
            elif isinstance(DBS_eff_act,float):
              comm.DBS_eff_act = bm.Variable(bm.array([DBS_eff_act]))
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
            comm.sorted_src_inds = jnp.sort(comm.src_inds)
            comm.dst_inds = comm.indices
            comm.sorted_dst_inds = jnp.sort(comm.dst_inds)
            comm.src_uniq = jnp.unique(src_inds)
            comm.dst_uniq = jnp.unique(comm.indices)
            if   isinstance(DBS_aff_act, list):
              for pop,pct in DBS_aff_act:
                if pop == v.pre:
                  comm.DBS_aff_act = bm.Variable(bm.array([pct]))
                  break
            elif isinstance(DBS_aff_act,float):
              comm.DBS_aff_act = bm.Variable(bm.array([DBS_aff_act]))
            if   isinstance(DBS_eff_act, list):
              for pop,pct in DBS_eff_act:
                if pop == v.pre:
                  comm.DBS_eff_act = bm.Variable(bm.array([pct]))
                  break
            elif isinstance(DBS_eff_act,float):
              comm.DBS_eff_act = bm.Variable(bm.array([DBS_eff_act]))

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
    self.last_dbs.value = jnp.take(self.dbs_times,self.dbs_idx-1)*(self.dbs_idx>0)
    def _pulse(connection):
      comm = connection.comm
      syn = connection.syn
      def pulse(x):
        syn.add_current(x*comm.weight)
      return pulse

    for tgt in self.tgts:
      # Where the target is post/destination so DBS will activate some
      # fraction of the afferent indices
      for post in self.posts[tgt]:
        comm = post.comm
        n_tgt = (comm.dst_uniq.size*jnp.abs(comm.DBS_aff_act)*(n_dbs>0)).astype(int)
        n_sel = jnp.arange(comm.dst_uniq.size)<n_tgt
        syns  = jnp.sort(jnp.where(n_sel,comm.dst_uniq,-1))
        
        x = jnp.zeros(post.post.size,dtype=int)
        def cond(args):
          i,j,x = args
          return jnp.any((i<comm.sorted_dst_inds.size) * (j<syns.size))
        def body(args):
          i,j,x = args
          v = syns.at[j].get()
          c = comm.sorted_dst_inds.at[i].get()
          x = x.at[c].set(x[c]+(c==v))
          i+=c<=v
          j+=c>v
          return i,j,x
        _,_,x = lax.while_loop(cond,body,(0,(~n_sel).sum(),x))
        x = x*n_dbs
        pulse = _pulse(post)
        pulse(x)
      for pre in self.pres[tgt]:
        comm = pre.comm
        n_tgt = (comm.src_uniq.size*jnp.abs(comm.DBS_eff_act)*(n_dbs>0)).astype(int)
        n_sel = jnp.arange(comm.src_uniq.size)<n_tgt
        syns  = jnp.sort(jnp.where(n_sel,comm.src_uniq,-1))
        
        x = jnp.zeros(pre.post.size,dtype=int)
        def cond(args):
          i,j,x = args
          return jnp.any((i<comm.sorted_src_inds.size) * (j<syns.size))
        def body(args):
          i,j,x = args
          v = syns.at[j].get()
          c = comm.sorted_src_inds.at[i].get()
          e = comm.sorted_dst_inds.at[i].get()
          x = x.at[e].set(x[e]+(c==v))
          i+=c<=v
          j+=c>v
          return i,j,x
        _,_,x = lax.while_loop(cond,body,(0,(~n_sel).sum(),x))
        x = x*n_dbs
        pulse = _pulse(pre)
        pulse(x)
    return self.net()
