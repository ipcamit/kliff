import libdescriptor as lds
from kliff.transforms.configuration_transforms import Descriptor
from kliff.transforms.configuration_transforms.default_hyperparams import symmetry_functions_set30


def test_lds_load():
    print(lds.AvailableDescriptors(0))
    assert lds != None

def test_descriptors():
    desc = Descriptor(cutoff=0.5, species=["Si"], descriptor="SymmetryFunctions",hyperparameters=symmetry_functions_set30())
    print(desc)
    assert False

