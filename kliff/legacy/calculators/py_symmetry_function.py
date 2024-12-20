import torch
# torch.set_default_dtype(torch.float64)
import math

@torch.jit.script
def cut_cos(r, rcut):
    """Cosine cutoff function."""
    if r < rcut:
        return 0.5 * (torch.cos(math.pi * r / rcut) + 1.0)
    else:
        return torch.tensor(0.0)

@torch.jit.script
def sym_g1(r, rcut):
    """Symmetry function G1."""
    phi = cut_cos(r, rcut)
    return phi

@torch.jit.script
def sym_g2(eta, Rs, r, rcut):
    """Symmetry function G2."""
    phi = torch.exp(-eta * (r - Rs) ** 2) * cut_cos(r, rcut)
    return phi

@torch.jit.script
def sym_g3(kappa, r, rcut):
    """Symmetry function G3."""
    phi = torch.cos(kappa * r) * cut_cos(r, rcut)
    return phi

@torch.jit.script
def sym_g4(zeta, lambda_, eta, rij, rik, rjk, rcut):
    """Symmetry function G4."""
    rcutij = rcut
    rcutik = rcut
    rcutjk = rcut

    if (rij > rcutij) or (rik > rcutik) or (rjk > rcutjk):
        return torch.tensor(0.0, device=rij.device)

    rijsq = rij ** 2
    riksq = rik ** 2 
    rjksq = rjk ** 2
    
    cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik)
    
    base = 1 + lambda_ * cos_ijk
    
    costerm = torch.pow(base, zeta) if base > 0 else torch.tensor(0.0)
    
    eterm = torch.exp(-eta * (rijsq + riksq + rjksq))

    phi = (
        torch.pow(torch.tensor(2.), 1. - zeta)
        * costerm
        * eterm
        * cut_cos(rij, rcutij)
        * cut_cos(rik, rcutik)
        * cut_cos(rjk, rcutjk)
    )
    return phi

@torch.jit.script
def sym_g5(zeta, lambda_, eta, rij, rik, rjk, rcut):
    """Symmetry function G5."""
    rcutij = rcut
    rcutik = rcut

    if (rij > rcutij) or (rik > rcutik):
        return torch.tensor(0.0, device=rij.device)

    rijsq, riksq, rjksq = rij ** 2, rik ** 2, rjk ** 2
    cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik)
    base = 1.0 + lambda_ * cos_ijk
    costerm = torch.pow(base, zeta) if base > 0 else torch.tensor(0.0, device=rij.device)
    eterm = torch.exp(-eta * (rijsq + riksq))

    phi = (
        torch.pow(torch.tensor(2.), 1. - zeta)
        * costerm
        * eterm
        * cut_cos(rij, rcutij)
        * cut_cos(rik, rcutik)
    )
    return phi


class BehlerSymmertryFunctions(torch.nn.Module):
    """
    Torch module for generating descriptors from dict, specifically for set 51 for right now.
    """
    def __init__(self, hyperparameters, cutoff):
        super().__init__()
        self.g2 = hyperparameters["g2"]
        self.g4 = hyperparameters["g4"]
        self.len = len(hyperparameters["g2"]) + len(hyperparameters["g4"])
        self.cutoff = cutoff
        self.cutoff_sq = cutoff * cutoff

    def forward(self, pos, numnei, neighs):
        n_contrib = numnei.size()[0]
        ptr = torch.cat([torch.tensor([0]), torch.cumsum(numnei, 0)])
        desc = torch.zeros(n_contrib, self.len)
        for i in range(n_contrib):
            pos_i = pos[i]

            nl = neighs[ptr[i]:ptr[i+1]]
            for jj in range(nl.size()[0]):
                j = nl[jj]
                pos_j = pos[j]
                r_ij = pos_j - pos_i
                r_ij_sq = torch.sum(r_ij * r_ij)
                if r_ij_sq > self.cutoff_sq:
                    continue
                r_ij_mag = torch.sqrt(r_ij_sq)

                # two body
                idx = 0
                for term in range(len(self.g2)):
                    eta = self.g2[term]["eta"]
                    Rs = self.g2[term]["Rs"]
                    desc[i, idx] += sym_g2(eta, Rs, r_ij_mag, self.cutoff)
                    idx += 1

                # three body
                for kk in range(jj + 1, nl.size()[0]):
                    k = nl[kk]
                    pos_k = pos[k]
                    r_ik = pos_k - pos_i
                    r_jk = pos_j - pos_k
                    r_ik_sq = torch.sum(r_ik * r_ik)
                    r_jk_sq = torch.sum(r_jk * r_jk)
                    if r_ik_sq > self.cutoff_sq:
                        continue
                    
                    r_ik_mag = torch.sqrt(r_ik_sq)
                    r_jk_mag = torch.sqrt(r_jk_sq)
                    # idx2 = 8
                    # print(len(self.g4))
                    # exit()
                    for term in range(len(self.g4)):
                        zeta = self.g4[term]["zeta"]
                        lambda_ = self.g4[term]["lambda"]
                        eta = self.g4[term]["eta"]
                        val = sym_g4(zeta, lambda_, eta, r_ij_mag, r_ik_mag, r_jk_mag, self.cutoff)
                        # if val ==0 
                        desc[i, term + 8] += val
                        # print(val.item(), r_ij_mag.item(), r_ik_mag.item(), r_jk_mag.item(), i, j.item(), k.item(), zeta, lambda_, eta)
                        # idx2 += 1
                    # print(idx)
        return desc


hyp = {'g2':
              [{'eta': torch.tensor(0.0035710676725828126), 'Rs': torch.tensor(1.3)},
               {'eta': torch.tensor(0.03571067672582813), 'Rs': torch.tensor(5.0)},
               {'eta': torch.tensor(0.07142135345165626), 'Rs': torch.tensor(2.0)},
               {'eta': torch.tensor(0.12498736854039845), 'Rs': torch.tensor(4.5)},
               {'eta': torch.tensor(0.21426406035496876), 'Rs': torch.tensor(2.5)},
               {'eta': torch.tensor(0.3571067672582813), 'Rs': torch.tensor(3.0)},
               {'eta': torch.tensor(0.7142135345165626), 'Rs': torch.tensor(4.0)},
               {'eta': torch.tensor(1.428427069033125), 'Rs': torch.tensor(3.5)}],
        'g4':
              [{'zeta':torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.00035710676725828126)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.00035710676725828126)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.00035710676725828126)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.00035710676725828126)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.010713203017748437)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.010713203017748437)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.010713203017748437)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.010713203017748437)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.0285685413806625)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.0285685413806625)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.0285685413806625)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.0285685413806625)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(4.0), 'lambda': -1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(4.0), 'lambda': 1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(16.0), 'lambda': -1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(16.0), 'lambda': 1, 'eta': torch.tensor(0.05356601508874219)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(4.0), 'lambda': -1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(4.0), 'lambda': 1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(16.0), 'lambda': -1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(16.0), 'lambda': 1, 'eta': torch.tensor(0.08927669181457032)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(4.0), 'lambda': -1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(4.0), 'lambda': 1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(16.0), 'lambda': -1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(16.0), 'lambda': 1, 'eta': torch.tensor(0.16069804526622655)},
               {'zeta': torch.tensor(1.0), 'lambda': -1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(1.0), 'lambda': 1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(2.0), 'lambda': -1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(2.0), 'lambda': 1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(4.0), 'lambda': -1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(4.0), 'lambda': 1, 'eta': torch.tensor(0.28568541380662504)},
               {'zeta': torch.tensor(16.0), 'lambda': 1, 'eta': torch.tensor(0.28568541380662504)}]
}

def get_desc():
    return BehlerSymmertryFunctions(hyp, 4.5)