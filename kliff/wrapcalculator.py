import collections
import os

import edn_format
import numpy as np


class WrapCalculator:
    """
    Wrapper over user-defined predictor function.

    Parameters
    ----------

    runner, function
      User defined function to generate prediction for a material property.
      For example, using LAMMPS to compute the elastic constant.

    outname, str
      Name of file that stores the results generated by 'runner'. The file should
      be in EDN format.

    keys: list of str
      Keywords in the 'outname' EDN file, whose value will be parsed as the prediction.
    """

    def __init__(self, params, outname, keys, runner, *args):
        self.params = params
        self.outname = outname
        self.keys = keys
        self.runner = runner
        self.args = args

    def get_prediction(self):
        """
        Return 1D array of floats.
        """
        self._update_params()
        self.runner(*self.args)
        return self._parse_edn(self.outname, self.keys)

    def _update_params(self):
        """Write parameters to file KIM_MODEL_PARAMS + modelname, and also give its path to the
        enviroment variable that has the same name: KIM_MODEL_PARAMS + modelname.
        """
        name = "KIM_MODEL_PARAMS_" + self.params._modelname
        self.params.echo_params(name, print_size=True)
        path = os.getcwd() + os.path.sep + name
        os.environ[name] = path

    def _parse_edn(self, fname, keys):
        """ Wrapper to use end_format to parse output file of 'runner' in edn format.

        Parameters
        ----------

        fname: str
          Name of the output file of OpenKIM test.

        keys: list of str
          Keyword in the edn format(think it as a dictionary), whose value will be returned.
        """

        with open(fname, "r") as fin:
            lines = fin.read()
        parsed = edn_format.loads(lines)
        values = []
        for k in keys:
            try:
                v = parsed[k]["source-value"]
            except KeyError:
                raise KeyError('Keyword "{}" not found in {}.'.format(k, self.outname))
            # make it a 1D array
            # we expect v as a python built-in object (if numpy object, this will fail)
            if isinstance(v, collections.Sequence):
                v = np.array(v).flatten()
            else:
                v = [v]
            values.append(v)
        return np.concatenate(values)
