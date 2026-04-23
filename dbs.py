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
    self.DBS_aff_act = bm.Variable(bm.array([DBS_aff_act]))
    self.DBS_eff_act = bm.Variable(bm.array([DBS_eff_act]))
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
            comm.src_indptr = comm.indptr.copy()
            comm.src_inds = src_inds
            comm.dst_inds = comm.indices
            comm.src_uniq = jnp.unique(src_inds)
            comm.dst_uniq = jnp.unique(comm.indices)
            comm.act_indices = bm.Variable(comm.indices)
            comm.act_indptr = bm.Variable(comm.indptr)
            comm.act_indptr = comm.act_indptr.at[:].set(v.post.size)
            comm.act_indptr = comm.act_indptr.at[0].set(0)

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
    self.DBS_aff_act.value -= 0.01*(n_dbs>0)
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
        n_tgt = (comm.dst_uniq.size*self.DBS_aff_act).astype(int)
        n_sel = jnp.arange(comm.dst_uniq.size)<n_tgt
        syns  = jnp.sort(jnp.where(n_sel,comm.dst_uniq,-1))

        mask = jnp.zeros_like(comm.sorted_dst_inds,dtype=bool)
        def cond(args):
          i,j,mask = args
          return jnp.all((i<comm.sorted_dst_inds.size) * (j<syns.size))
        def body(args):
          i,j,mask = args
          v = syns.at[j].get()
          c = comm.sorted_dst_inds.at[i].get()
          mask = mask.at[i].set(c==v)
          i+=c<=v
          j+=c>v
          return i,j,mask
        _,_,mask = lax.while_loop(cond,body,(0,0,mask))

        # mask = jnp.isin(comm.sorted_dst_inds,syns)
          
        # indices = jnp.where(mask,comm.dst_inds,-1)
        # x = jnp.zeros(post.post.size)
        # for i in range(post.post.size[0]):
        #   ind = indices[i]
        #   x = x.at[ind].set(x[ind] + ind>=0)
        indices = mask * comm.sorted_dst_inds
        x = jnp.bincount(indices,length=post.post.size[0])
        x = x.at[0].set( x[0]-(~mask).sum() )
        x = x*n_dbs
        pulse = _pulse(post)
        pulse(x)
      # for pre in self.pres[tgt]:
      #   x = jnp.zeros(pre.pre.size)
      #   x = x.at[0].set(n_dbs>0)
      #   comm = pre.comm
      #   n_tgt = (comm.src_uniq.size*self.DBS_eff_act).astype(int)
      #   n_sel = jnp.arange(comm.src_uniq.size)<n_tgt
      #   syns  = jnp.where(n_sel,comm.src_uniq,-1)
      #   mask  = jnp.isin(comm.src_inds,syns)
      #   indices = jnp.sort(jnp.where(mask,comm.dst_inds,0),descending=True)
      #   n_syns = jnp.sum(mask)
      #   indptr = jnp.ones_like(comm.indptr)*pre.post.size[0]
      #   indptr = indptr.at[0].set(0)
      #   indptr = indptr.at[1].set(n_syns)
      #   pulse = _pulse(pre)
      #   pulse(x,indices,indptr)
    return self.net()
