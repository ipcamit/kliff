import copy

import numpy as np
from loguru import logger

from kliff.utils import split_string


class Fisher:
    r"""
    Fisher information matrix.

    Compute the Fisher information according to

    ..math::
        I_{ij} = \sum_m \frac{\partial \bm f_m}{\partial \theta_i}
        \cdot \frac{\partial \bm f_m}{\partial \theta_j}

    where :math:`f_m` are the forces on atoms in configuration :math:`m`, :math:`\theta_i`
    is the ith model parameter.
    Derivatives are computed numerically using Ridders' algorithm:
    https://en.wikipedia.org/wiki/Ridders%27_method

    Parameters
    ----------
    model:
        A model object.
    """

    def __init__(self, model, configurations):
        self.model = model
        self.F = None
        self.F_stdev = None
        self.delta_params = None
        self.configurations = configurations

    def run(self, verbose=1):
        """
        Compute the Fisher information matrix and the standard deviation.

        Parameters
        ----------
        verbose: int
            If ``0``, do not write out to file; if ``1``, write to a file named
            ``analysis_Fisher_info_matrix.txt``.

        Returns
        -------
        I: 2D array, shape(N, N)
            Fisher information matrix, where N is the number of parameters.

        I_stdev: 2D array, shape(N, N)
            Standard deviation of Fisher information matrix, where N is the number of
            parameters.
        """

        logger.info("Start computing Fisher information matrix.")

        I_all = []

        # cas = self.model.get_compute_arguments()
        for i, ca in enumerate(self.configurations):
            if i % 100 == 0:
                logger.info(f"Processing configuration {i}.")

            dfdp = self._compute_jacobian_one_config(ca)
            I_all.append(np.dot(dfdp.T, dfdp))
        I = np.mean(I_all, axis=0)
        I_stdev = np.std(I_all, axis=0)

        self._write_result(I, I_stdev, verbose)
        logger.info("Finish computing Fisher information matrix.")

        return I, I_stdev

    def _write_result(self, I, stdev, verbose, path="analysis_Fisher_info_matrix.txt"):

        params = self.model.get_opt_params()
        nparams = len(params)
        names = []
        values = []
        component_idx = []
        for i in range(len(params)):
            out = self.model.get_opt_param_name_value_and_indices(i)
            n, v, p_idx, c_idx = out
            names.append(n)
            values.append(v)
            component_idx.append(c_idx)

        # header
        header = "#" * 80 + "\n# Fisher information matrix.\n#\n"
        msg = (
            "The size of the parameter list is {0}, and thus the Fisher information "
            "matrix is a {0} by {0} matrix. The rows (columns) are associated with the "
            "parameters in the following order:".format(nparams)
        )
        header += split_string(msg, length=80, starter="#")
        header += "#\n"
        header += (
            "row (column) index   param name    param value    param component index\n"
        )
        for i, (n, v, c) in enumerate(zip(names, values, component_idx)):
            header += "{}    {}    {:23.15e}    {}\n".format(i, n, v, c)
        header += "#" * 80 + "\n"
        print(header)

        # write to file
        if verbose > 0:
            with open(path, "w") as fout:

                fout.write(header)

                fout.write(
                    "\n# Fisher information matrix, shape({0}, {0})\n".format(nparams)
                )
                for line in I:
                    for v in line:
                        fout.write("{:23.15e} ".format(v))
                    fout.write("\n")

                fout.write(
                    "\n# Standard deviation in Fisher information matrix, "
                    "shape({0}, {0})\n".format(nparams)
                )
                for line in stdev:
                    for v in line:
                        fout.write("{:23.15e} ".format(v))
                    fout.write("\n")

    def _compute_jacobian_one_config(self, ca):
        """
        Compute the Jacobian of forces w.r.t. parameters for one configuration.

        Parameters
        ----------
        ca: object
            `compute argument` associated with one configuration.
        """

        try:
            import numdifftools as nd
        except ImportError as e:
            raise ImportError(
                '{}\nFisher information analyzer needs "numdifftools". Please install '
                "it first.".format(str(e))
            )

        # compute Jacobian of forces w.r.t. parameters
        original_params = self.model.get_opt_params()
        Jfunc = nd.Jacobian(self._compute_forces_one_config)
        j = Jfunc(copy.deepcopy(original_params), ca)

        # restore params back
        self.model.update_model_params(original_params)

        return j

    def _compute_forces_one_config(self, params, ca):
        """
        Compute forces using a specific set of model parameters.

        Parameters
        ----------
        params: list
          the parameter values

        ca: object
            `compute argument` associated with one configuration

        Return
        ------
        forces: 1D array
            the forces on atoms in this configuration
        """
        self.model.update_model_params(params)
        forces = self.model(ca, compute_forces=True)["forces"]
        # forces = self.model.get_forces(ca)
        forces = np.reshape(forces, (-1,))

        return forces
