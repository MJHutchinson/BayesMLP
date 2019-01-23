"""
  Harness for calling a function.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=no-member
# pylint: disable=relative-import
# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local
from utils.general_utils import map_to_cube, map_to_bounds
# from domains import EuclideanDomain

EVAL_ERROR_CODE = 'eval_error_2401181243'


class FunctionCaller(object):
  """ The basic function caller class.
      All other function callers should inherit this class.
  """

  def __init__(self, func, domain, opt_pt=None, opt_val=None, noise_type='none',
               noise_params=None, descr=''):
    """ Constructor.
      - func: takes argument x and returns function value.
      - noise_type: A string indicating what type of noise to add. If 'none' will not
                    add any noise.
      - noise_params: Any parameters for the noise random variable.
      - raw_opt_pt, raw_opt_val: optimum point and value if known. "raw" because, when
          actually calling the function, you might want to do some normalisation of x.
    """
    self.func = func
    self.domain = domain
    self.noise_type = noise_type
    self.noise_params = noise_params
    self.descr = descr
    self.opt_pt = opt_pt     # possibly over-written by child class
    self.opt_val = opt_val   # possibly over-written by child class
    self.noise_adder = None
    self._set_up_noise_adder()

  def _set_up_noise_adder(self):
    """ Sets up a function to add noise. """
    if self.noise_type == 'none':
      self.noise_adder = lambda num_samples: np.zeros(shape=(num_samples))
    elif self.noise_type == 'gauss':
      self.noise_adder = lambda num_samples: np.random.normal(size=(num_samples))
    else:
      raise NotImplementedError(('Not implemented %s yet. Only implemented Gaussian noise'
                                 + ' so far.')%(self.noise_type))

  def eval_single(self, x, qinfo=None, noisy=True):
    """ Evaluates func at a single point x. If noisy is True and noise_type is \'none\'
        will add noise.
    """
    qinfo = Namespace() if qinfo is None else qinfo
    true_val = float(self.func(x))
    if true_val == EVAL_ERROR_CODE:
      val = EVAL_ERROR_CODE
    else:
      val = true_val if not noisy else true_val + self.noise_adder(1)[0]
    # put everything into qinfo
    qinfo.point = x
    qinfo.true_val = true_val
    qinfo.val = val
    return val, qinfo

  def eval_multiple(self, X, qinfos=None, noisy=True):
    """ Evaluates the function at a list of points in a list X.
        Creating this because when the domain is Euclidean there may be efficient
        vectorised implementations.
    """
    qinfos = [None] * len(X) if qinfos is None else qinfos
    ret_vals = []
    ret_qinfos = []
    for i in range(len(X)):
      val, qinfo = self.eval_single(X[i], qinfos[i], noisy)
      ret_vals.append(val)
      ret_qinfos.append(qinfo)
    return ret_vals, ret_qinfos
