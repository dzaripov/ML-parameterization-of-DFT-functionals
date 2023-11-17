# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""An interface to DM21 family of exchange-correlation functionals for PySCF."""

import enum
from typing import Generator, Optional, Sequence, Tuple, Union

import attr
import numpy as np
from pyscf import dft
from pyscf import gto
from pyscf.dft import numint
import torch


from density_functional_approximation_dm21.functional import NN_FUNCTIONAL

# TODO(b/196260242): avoid depending upon private function
_dot_ao_ao = numint._dot_ao_ao  # pylint: disable=protected-access
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
np.random.seed(42)


@enum.unique
class Functional(enum.Enum):
  """Enum for exchange-correlation functionals in the DM21 family.

  Attributes:
    DM21: trained on molecules dataset, and fractional charge, and fractional
      spin constraints.
    DM21m: trained on molecules dataset.
    DM21mc: trained on molecules dataset, and fractional charge constraints.
    DM21mu: trained on molecules dataset, and electron gas constraints.
  """
  # Break pylint's preferred naming pattern to match the functional names used
  # in the paper.
  # pylint: disable=invalid-name
  DM21 = enum.auto()
  NN_PBE = enum.auto()
  NN_XALPHA = enum.auto()
  # pylint: enable=invalid-name


@attr.s(auto_attribs=True)
class _GridState:
  """Internal state required for the numerical grid.

  Attributes:
    coords: coordinates of the grid. Shape (N, 3), where N is the number of grid
      points.
    weight: weight associated with each grid point. Shape (N).
    mask: mask indicating whether a shell is zero at a grid point. Shape
      (N, nbas) where nbas is the number of shells in the basis set. See
      pyscf.dft.gen_grids.make_mask.
    ao: atomic orbitals evaluated on the grid. Shape (N, nao), where nao is the
      number of atomic orbitals, or shape (:, N, nao), where the 0-th element
      contains the ao values, the next three elements contain the first
      derivatives, and so on.
  """
  coords: np.ndarray
  weight: np.ndarray
  mask: np.ndarray
  ao: np.ndarray


@attr.s(auto_attribs=True)
class _SystemState:
  """Internal state required for system of interest.

  Attributes:
    mol: PySCF molecule
    dms: density matrix or matrices (unrestricted calculations only).
      Restricted calculations: shape (nao, nao), where nao is the number of
      atomic orbitals.
      Unrestricted calculations: shape (2, nao, nao) or a sequence (length 2) of
      arrays of shape (nao, nao), and dms[0] and dms[1] are the density matrices
      of the alpha and beta electrons respectively.
  """
  mol: gto.Mole
  dms: Union[np.ndarray, Sequence[np.ndarray]]


def _get_number_of_density_matrices(dms):
  """Returns the number of density matrices in dms."""
  # See pyscf.numint.NumInt._gen_rho_evaluator
  if isinstance(dms, np.ndarray) and dms.ndim == 2:
    return 1
  return len(dms)


class NeuralNumInt(numint.NumInt):
  """A wrapper around pyscf.dft.numint.NumInt for the DM21 functionals.

  In order to supply the local Hartree-Fock features required for the DM21
  functionals, we lightly wrap the NumInt class. The actual evaluation of the
  exchange-correlation functional is performed in NeuralNumInt.eval_xc.

  Usage:
    mf = dft.RKS(...)  # dft.ROKS and dft.UKS are also supported.
    # Specify the functional by monkey-patching mf._numint rather than using
    # mf._xc or mf._define_xc_.
    mf._numint = NeuralNumInt(Functional.DM21)
    mf.kernel()
  """

  def __init__(self, functional):
    """Constructs a NeuralNumInt object.

    Args:
      model_class = class of NN torch model
      model_path = path to model's state_dict
    """
    #change this
    self._functional_name = functional.name
    self._model = NN_FUNCTIONAL(self._functional_name)
    #to our NN models

    # All DM21 functionals use local Hartree-Fock features with a non-range
    # separated 1/r kernel and a range-seperated kernel with \omega = 0.4.
    # Note an omega of 0.0 is interpreted by PySCF and libcint to indicate no
    # range-separation.

    self._grid_state = None
    self._system_state = None
    super().__init__()

  # DM21* functionals include the hybrid term directly, so set the
  # range-separated and hybrid parameters expected by PySCF to 0 so PySCF
  # doesn't also add these contributions in separately.
  def rsh_coeff(self, *args):
    """Returns the range separated parameters, omega, alpha, beta."""
    return [0.0, 0.0, 0.0]

  def hybrid_coeff(self, *args, **kwargs):
    """Returns the fraction of Hartree-Fock exchange to include."""
    return 0.0

  def _xc_type(self, *args, **kwargs):
    return 'MGGA'

  def torch_grad(self, outputs, inputs):
    grads = torch.autograd.grad(outputs, inputs,
                                 create_graph=True,
                                 only_inputs=True,
                                 )
    return grads

  def model_predict(self, features):
    """Compute model's prediction and gradients of neural network
    """

    vxc, vrho, vsigma, vtau = self._model(features, device)

    # The potential is the local exchange correlation divided by the
    # total density. Add a small constant to deal with zero density.
    self._vxc = vxc

    # The derivatives of the exchange-correlation (XC) energy with respect to
    # input features.  PySCF weights the (standard) derivatives by the grid
    # weights, so we need to compute this with respect to the unweighted sum
    # over grid points.
    self._vrho = vrho
    self._vsigma = vsigma
    self._vtau = vtau
    # Standard meta-GGAs do not have a dependency on local HF, so we need to
    # compute the contribution to the Fock matrix ourselves. Just use the
    # weighted XC energy to avoid having to weight this later.


    outputs = {
        'vxc': self._vxc.detach().cpu().numpy(),
        'vrho': torch.stack(self._vrho).detach().cpu().numpy(),
        'vsigma': torch.stack(self._vsigma).detach().cpu().numpy(),
        'vtau': torch.stack(self._vtau).detach().cpu().numpy(),
    }
    return outputs
  
  def model_predict_old(self, features, device):
    """Compute model's prediction and gradients of neural network
    """

    local_xc, vxc, input_features = self._model(features, device)
    unweighted_xc = torch.sum(local_xc, dim=1)

    # The potential is the local exchange correlation divided by the
    # total density. Add a small constant to deal with zero density.
    self._vxc = vxc

    # The derivatives of the exchange-correlation (XC) energy with respect to
    # input features.  PySCF weights the (standard) derivatives by the grid
    # weights, so we need to compute this with respect to the unweighted sum
    # over grid points.
    self._vrho = self.torch_grad(
        unweighted_xc, [input_features['rho_a'], input_features['rho_b']])
    self._vsigma = self.torch_grad(
        unweighted_xc, [
            input_features['norm_grad_a'], input_features['norm_grad_b'],
            input_features['norm_grad']
        ])
    self._vtau = self.torch_grad(
        unweighted_xc, [input_features['tau_a'], input_features['tau_b']])

    # Standard meta-GGAs do not have a dependency on local HF, so we need to
    # compute the contribution to the Fock matrix ourselves. Just use the
    # weighted XC energy to avoid having to weight this later.

    outputs = {
        'vxc': self._vxc.detach().cpu().numpy(),
        'vrho': torch.stack(self._vrho).detach().cpu().numpy(),
        'vsigma': torch.stack(self._vsigma).detach().cpu().numpy(),
        'vtau': torch.stack(self._vtau).detach().cpu().numpy(),
    }
    return outputs 

  def nr_uks(self,
             mol: gto.Mole,
             grids: dft.Grids,
             xc_code: str,
             dms: Union[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]],
             relativity: int = 0,
             hermi: int = 0,
             max_memory: float = 20000,
             verbose=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """Calculates UKS XC functional and potential matrix on a given grid.

    Args:
      mol: PySCF molecule.
      grids: grid on which to evaluate the functional.
      xc_code: XC code. Unused. NeuralNumInt hard codes the XC functional
        based upon the functional argument given to the constructor.
      dms: the density matrix or sequence of density matrices for each spin
        channel. Multiple density matrices for each spin channel are not
        currently supported. Each density matrix is shape (nao, nao), where nao
        is the number of atomic orbitals.
      relativity: Unused. (pyscf.dft.numint.NumInt.nr_rks does not currently use
        this argument.)
      hermi: 0 if the density matrix is Hermitian, 1 if the density matrix is
        non-Hermitian.
      max_memory: the maximum cache to use, in MB.
      verbose: verbosity level. Unused. (PySCF currently does not handle the
        verbosity level passed in here.)

    Returns:
      nelec, excsum, vmat, where
        nelec is the number of alpha, beta electrons obtained by numerical
        integration of the density matrix as an array of size 2.
        excsum is the functional's XC energy.
        vmat is the functional's XC potential matrix, shape (2, nao, nao), where
        vmat[0] and vmat[1] are the potential matrices for the alpha and beta
        spin channels respectively.

    Raises:
      NotImplementedError: if multiple density matrices for each spin channel
      are supplied.
    """
    # Wrap nr_uks so we can store internal variables required to evaluate the
    # contribution to the XC potential from local Hartree-Fock features.
    # See pyscf.dft.numint.nr_uks for more details.
    if isinstance(dms, np.ndarray) and dms.ndim == 2:  # RHF DM
      ndms = _get_number_of_density_matrices(dms)
    else:
      ndms = _get_number_of_density_matrices(dms[0])
    if ndms > 1:
      raise NotImplementedError(
          'NeuralNumInt does not support multiple density matrices. '
          'Only ground state DFT calculations are currently implemented.')
    
    nao = mol.nao_nr()
    self._system_state = _SystemState(mol=mol, dms=dms)
    nelec, excsum, vmat = super().nr_uks(
        mol=mol,
        grids=grids,
        xc_code=xc_code,
        dms=dms,
        relativity=relativity,
        hermi=hermi,
        max_memory=max_memory,
        verbose=verbose)

    # Clear internal state to prevent accidental re-use.
    self._system_state = None
    self._grid_state = None
    return nelec, excsum, vmat

  def block_loop(
      self,
      mol: gto.Mole,
      grids: dft.Grids,
      nao: Optional[int] = None,
      deriv: int = 0,
      max_memory: float = 2000,
      non0tab: Optional[np.ndarray] = None,
      blksize: Optional[int] = None,
      buf: Optional[np.ndarray] = None
  ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None,
                 None]:
    """Loops over the grid by blocks. See pyscf.dft.numint.NumInt.block_loop.

    Args:
      mol: PySCF molecule.
      grids: grid on which to evaluate the functional.
      nao: number of basis functions. If None, obtained from mol.
      deriv: unused. The first functional derivatives are always computed.
      max_memory: the maximum cache to use for the information on the grid, in
        MB. Determines the size of each block if blksize is None.
      non0tab: mask determining if a shell in the basis set is zero at a grid
        point. Shape (N, nbas), where N is the number of grid points and nbas
        the number of shells in the basis set. Obtained from grids if not
        supplied.
      blksize: size of each block. Calculated from max_memory if None.
      buf: buffer to use for storing ao. If None, a new array for ao is created
        for each block.

    Yields:
      ao, mask, weight, coords: information on a block of the grid containing N'
      points, where
        ao: atomic orbitals evaluated on the grid. Shape (N', nao), where nao is
        the number of atomic orbitals.
        mask: mask indicating whether a shell in the basis set is zero at a grid
        point. Shape (N', nbas).
        weight: weight associated with each grid point. Shape (N').
        coords: coordinates of the grid. Shape (N', 3).
    """
    # Wrap block_loop so we can store internal variables required to evaluate
    # the contribution to the XC potential from local Hartree-Fock features.
    for ao, mask, weight, coords in super().block_loop(
        mol=mol,
        grids=grids,
        nao=nao,
        deriv=deriv,
        max_memory=max_memory,
        non0tab=non0tab,
        blksize=blksize,
        buf=buf):
      # Cache the curent block so we can access it in eval_xc.
      self._grid_state = _GridState(
          ao=ao, mask=mask, weight=weight, coords=coords)
      yield ao, mask, weight, coords

  def construct_functional_inputs(
      self,
      mol: gto.Mole,
      dms: Union[np.ndarray, Sequence[np.ndarray]],
      spin: int,
      coords: np.ndarray,
      weights: np.ndarray,
      rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      ao: Optional[np.ndarray] = None,
  ) -> Tuple[dict, Tuple[np.ndarray, np.ndarray]]:
    """Constructs the input features required for the functional.

    Args:
      mol: PySCF molecule.
      dms: density matrix of shape (nao, nao) (restricted calculations) or of
        shape (2, nao, nao) (unrestricted calculations) or tuple of density
        matrices for each spin channel, each of shape (nao, nao) (unrestricted
        calculations).
      spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
        spin-polarized (unrestricted) otherwise.
      coords: coordinates of the grid. Shape (N, 3), where N is the number of
        grid points.
      weights: weight associated with each grid point. Shape (N).
      rho: density and density derivatives at each grid point. Single array
        containing the total density for restricted calculations, tuple of
        arrays for each spin channel for unrestricted calculations. Each array
        has shape (6, N). See pyscf.dft.numint.eval_rho and comments in
        FunctionalInputs for more details.
      ao: The atomic orbitals evaluated on the grid, shape (N, nao). Computed if
        not supplied.

    Returns:
      inputs, fxx, where
        inputs: FunctionalInputs object containing the inputs (as np.ndarrays)
        for the functional.
        fxx: intermediates, shape (N, nao) for the alpha- and beta-spin
        channels, required for computing the first derivative of the local
        Hartree-Fock density with respect to the density matrices. See
        compute_hfx_density for more details.
    """
    if spin == 0:
      # RKS
      rhoa = rho / 2
      rhob = rho / 2
    else:
      # UKS
      rhoa, rhob = rho


    return {'rho_a':torch.from_numpy(rhoa).type(torch.float32),
        'rho_b':torch.from_numpy(rhob).type(torch.float32),
        'grid_weights':torch.from_numpy(weights)[:, None].type(torch.float32)}

  def eval_xc(
      self,
      xc_code: str,
      rho: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
      spin: int = 0,
      relativity: int = 0,
      deriv: int = 1,
      verbose=None,
      omega=None
  ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             None, None]:
    """Evaluates the XC energy and functional derivatives.

    See pyscf.dft.libxc.eval_xc for more details on the interface.

    Note: this also sets self._vmat_extra, which contains the contribution the
    the potential matrix from the local Hartree-Fock terms in the functional.

    Args:
      xc_code: unused.
      rho: density and density derivatives at each grid point. Single array
        containing the total density for restricted calculations, tuple of
        arrays for each spin channel for unrestricted calculations. Each array
        has shape (6, N), where N is the number of grid points. See
        pyscf.dft.numint.eval_rho and comments in FunctionalInputs for more
        details.
      spin: 0 for a spin-unpolarized (restricted Kohn-Sham) calculation, and
        spin-polarized (unrestricted) otherwise.
      relativity: unused.
      deriv: unused. The first functional derivatives are always computed.
      verbose: unused.

    Returns:
      exc, vxc, fxc, kxc, where:
        exc is the exchange-correlation potential matrix evaluated at each grid
        point, shape (N).
        vxc is (vrho, vgamma, vlapl, vtau), the first-order functional
        derivatives evaluated at each grid point, each shape (N).
        fxc is set to None. (The second-order functional derivatives are not
        computed.)
        kxc is set to None. (The third-order functional derivatives are not
        computed.)
    """
    del xc_code, verbose, relativity, deriv  # unused

    # Retrieve cached state.
    ao = self._grid_state.ao
    if ao.ndim == 3:
      # Just need the AO values, not the gradients.
      ao = ao[0]
    if self._grid_state.weight is None:
      weights = np.array([1.])
    else:
      weights = self._grid_state.weight
    mask = self._grid_state.mask

    features = self.construct_functional_inputs(
        mol=self._system_state.mol,
        dms=self._system_state.dms,
        spin=spin,
        rho=rho,
        weights=weights,
        coords=self._grid_state.coords,
        ao=ao)
    
    grads = self.model_predict_old(features, device)
    exc, vrho, vsigma, vtau = grads['vxc'], grads['vrho'], grads['vsigma'], grads['vtau']

    
    mol = self._system_state.mol
    shls_slice = (0, mol.nbas)
    ao_loc_nr = mol.ao_loc_nr()
    # Note: tf.gradients returns a list of gradients.
    # vrho, vsigma, vtau are derivatives of objects that had
    # tf.expand_dims(..., 1) applied. The [:, 0] indexing undoes this by
    # selecting the 0-th (and only) element from the second dimension.

    if spin == 0:
      vxc_0 = (vrho[0][0, :] + vrho[1][0, :]) / 2.
      # pyscf expects derivatives with respect to:
      # grad_rho . grad_rho.
      # The functional uses the first and last as inputs, but then has
      # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
      # as input. The following computes the correct total derivatives.
      vxc_1 = (vsigma[0][0, :] / 4. + vsigma[1][0, :] / 4. + vsigma[2][:, 0])
      vxc_3 = (vtau[0][0, :] + vtau[1][0, :]) / 2.
      vxc_2 = np.zeros_like(vxc_3)

    else:
      vxc_0 = np.stack([vrho[0][0, :], vrho[1][0, :]], axis=1)
      # pyscf expects derivatives with respect to:
      # grad_rho_a . grad_rho_a
      # grad_rho_a . grad_rho_b
      # grad_rho_b . grad_rho_b
      # The functional uses the first and last as inputs, but then has
      # grad_(rho_a + rho_b) . grad_(rho_a + rho_b)
      # as input. The following computes the correct total derivatives.
      vxc_1 = np.stack([
          vsigma[0][0, :] + vsigma[2][0, :], 2. * vsigma[2][0, :],
          vsigma[1][0, :] + vsigma[2][0, :]
      ],
                       axis=1)
      vxc_3 = np.stack([vtau[0][0, :], vtau[1][0, :]], axis=1)
      vxc_2 = np.zeros_like(vxc_3)


    fxc = None  # Second derivative not implemented
    kxc = None  # Second derivative not implemented
    exc = exc.astype(np.float64)
    vxc = tuple(v.astype(np.float64) for v in (vxc_0, vxc_1, vxc_2, vxc_3))
    return exc, vxc, fxc, kxc
