import sys
from MDAnalysis import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# python3 -u train.py active.gro active.xtc topol.top
# ^ command to run file - needs active.gro, active.xtc and topol.top
# GitHub repo has link to xtc file amd the repo itself has the other two files

gro_file = str(sys.argv[-3])
xtc_file = str(sys.argv[-2])

### CONFIG PARAMETERS ###
rel_atoms = ['N','C','CA','CB','O'] ### atoms to consider from data
universe = Universe(gro_file, xtc_file) ### files to take as reference
batch_size = 4 ### batch size


topol_file = str(sys.argv[-1])
topol = open(topol_file,'r')
lines = topol.readlines()

bonds_begin = 0
pairs_begin = 0
angles_begin = 0
dihed_begin1 = 0
dihed_begin2 = 0
for i in range(len(lines)):
    if lines[i] == '[ bonds ]\n':
        bonds_begin = i
    if lines[i] == '[ pairs ]\n':
        pairs_begin = i
    if lines[i] == '[ angles ]\n':
        angles_begin = i
    if lines[i] == '[ dihedrals ]\n':
        if dihed_begin1 == 0:
            dihed_begin1 = i
        else:
            dihed_begin2 = i
print(bonds_begin, pairs_begin, angles_begin, dihed_begin1, dihed_begin2)

bonds = []
for i in range(bonds_begin+2, pairs_begin):
    if lines[i] == '\n':
        break
    else:
        nums = lines[i].split()[:-1]
        nums = [int(i) for i in nums]
        bonds.append(nums)
pairs = []
for i in range(pairs_begin+2, angles_begin):
    if lines[i] == '\n':
        break
    else:
        nums = lines[i].split()[:-1]
        nums = [int(i) for i in nums]
        pairs.append(nums)
angles = []
for i in range(angles_begin+2, dihed_begin1):
    if lines[i] == '\n':
        break
    else:
        nums = lines[i].split()[:-1]
        nums = [int(i) for i in nums]
        angles.append(nums)
diheds = []
for i in range(dihed_begin1+2, dihed_begin2):
    if lines[i] == '\n':
        break
    else:
        nums = lines[i].split()[:-1]
        nums = [int(i) for i in nums]
        diheds.append(nums)
diheds_common = []
for i in range(dihed_begin2+2, len(lines)):
    if lines[i] == '\n':
        break
    else:
        nums = lines[i].split()[:-1]
        nums = [int(i) for i in nums]
        diheds_common.append(nums)

all_atoms = universe.atoms.names
rel_atoms_universe = []
rel_atoms_universe_names = []
for i in range(len(all_atoms)):
    if(all_atoms[i] in rel_atoms):
        rel_atoms_universe.append(i+1)
        rel_atoms_universe_names.append(all_atoms[i])

all_coords = []
for frame in universe.trajectory:
    req_atoms_in_frame = []
    all_atoms_in_frame = universe.atoms.positions
    for j in rel_atoms_universe:
        req_atoms_in_frame.append(all_atoms_in_frame[j])
    all_coords.append(req_atoms_in_frame)
all_coords = np.array(all_coords)
all_coords = np.transpose(all_coords, (0, 2, 1))

class dataset(Dataset):
    
    def __init__(self, x_train, transform=True):
        self.transform = transform
        self.x = torch.from_numpy(x_train)
        self.x = self.x.to(torch.float32)
        
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
x_train, x_test = train_test_split(all_coords, test_size = 0.2, random_state = 42)
train_dataset = dataset(x_train)
test_dataset = dataset(x_test)
train_loader = DataLoader(train_dataset, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size = batch_size)

for i, data in enumerate(train_loader):
    continue
    
rel_bonds = []
rel_pairs = []
rel_angles = []
rel_diheds = []
rel_diheds_common = []
for bond in bonds:
    if all(item in rel_atoms_universe for item in bond):
        rel_bonds.append(bond)
for pair in pairs:
    if all(item in rel_atoms_universe for item in pair):
        rel_pairs.append(pair)
for angle in angles:
    if all(item in rel_atoms_universe for item in angle):
        rel_angles.append(angle)
for dihed in diheds:
    if all(item in rel_atoms_universe for item in dihed):
        rel_diheds.append(dihed)
for dihed_common in diheds_common:
    if all(item in rel_atoms_universe for item in dihed_common):
        rel_diheds_common.append(dihed_common)
        
named_bonds = []
for bond in rel_bonds:
    atom1 = rel_atoms_universe.index(bond[0])
    atom2 = rel_atoms_universe.index(bond[1])
    named_bonds.append(rel_atoms_universe_names[atom1] + ' - ' + rel_atoms_universe_names[atom2])
unique_named_bonds = list(set(named_bonds))
print(unique_named_bonds)
k_b_vals = [469.0, 469.0, 337.0, 570.0, 469.0, 490.0]
r_vals = [1.4040, 1.4040, 1.4490, 1.2290, 1.4090, 1.3350]
def bonded_loss(frames, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds):
    pots = []
    for frame in frames:
        coords = torch.transpose(frame, 0, 1)
        dists = []
        for bond in rel_bonds:
            coord1 = coords[rel_atoms_universe.index(bond[0])]
            coord2 = coords[rel_atoms_universe.index(bond[1])]
            dist = torch.norm(coord1-coord2)
            dists.append(dist)
        pot = 0
        for i in range(len(named_bonds)):
            ind = unique_named_bonds.index(named_bonds[i])
            k = k_b_vals[ind]
            r = r_vals[ind]
            pot = pot + (k * (r - dists[i]) ** 2)
        pots.append(pot.cpu().detach())
    pots = np.array(pots)
    return pots

named_angles = []
for angle in rel_angles:
    atom1 = rel_atoms_universe.index(angle[0])
    atom2 = rel_atoms_universe.index(angle[1])
    atom3 = rel_atoms_universe.index(angle[2])
    named_angles.append(rel_atoms_universe_names[atom1] + ' - ' + rel_atoms_universe_names[atom2] \
                        + ' - ' + rel_atoms_universe_names[atom3])
unique_named_angles = list(set(named_angles))
print(unique_named_angles)
k_theta_vals = [70.0, 70.0, 63.0, 50.0, 63.0, 80.0, 80.0]
theta_vals = [123.50, 116.60, 110.10, 121.90, 110.10, 122.90, 120.40]
def theta_loss(frames, rel_atoms_universe, rel_angles, named_angles, unique_named_angles):
    pots = []
    for frame in frames:
        coords = torch.transpose(frame, 0, 1)
        angles = []
        for angle in rel_angles:
            coord1 = coords[rel_atoms_universe.index(angle[0])]
            coord2 = coords[rel_atoms_universe.index(angle[1])]
            coord3 = coords[rel_atoms_universe.index(angle[2])]
            vec1 = coord1 - coord2
            vec2 = coord3 - coord2
            dot_product = torch.dot(vec1, vec2)
            norm_vec1 = torch.norm(vec1)
            norm_vec2 = torch.norm(vec2)
            angle_radians = torch.atan2(torch.norm(torch.cross(vec1, vec2)), torch.dot(vec1, vec2))
            angles.append(angle_radians)
        pot = 0
        for i in range(len(named_angles)):
            ind = unique_named_angles.index(named_angles[i])
            k = k_theta_vals[ind]
            theta = theta_vals[ind] * (np.pi / 180)
            pot = pot + (k * (theta - angles[i]) ** 2)
        pots.append(pot.cpu().detach())
    pots = np.array(pots)
    return pots

named_diheds = []
for dihed in rel_diheds:
    atom1 = rel_atoms_universe.index(dihed[0])
    atom2 = rel_atoms_universe.index(dihed[1])
    atom3 = rel_atoms_universe.index(dihed[2])
    atom4 = rel_atoms_universe.index(dihed[3])
    named_diheds.append(rel_atoms_universe_names[atom1] + ' - ' + rel_atoms_universe_names[atom2] \
                        + ' - ' + rel_atoms_universe_names[atom3] + ' - ' + rel_atoms_universe_names[atom4])
unique_named_diheds = list(set(named_diheds))
print(unique_named_diheds)
v_vals = [10.00, [0.00, 0.55, 1.58, 0.45], 0.00, 14.50, 14.50, [0.00, 0.42, 0.27, 0.00], 10.00, 14.50]
n_vals = [2.0, [4.0, 3.0, 2.0, 1.0], 2.0, 2.0, 2.0, [4.0, 3.0, 2.0, 1.0], 2.0, 2.0]
gamma_vals = [180.0, [0.0, 180.0, 180.0, 180.0], 0.0, 180.0, 180.0, [0.0, 0.0, 0.0, 0.0], 180.0, 180.0]
divider_vals = [4.0, [1.0, 1.0, 1.0, 1.0], 6.0, 4.0, 4.0, [1.0, 1.0, 1.0, 1.0], 4.0, 4.0]
def dihed_loss(frames, rel_atoms_universe, rel_diheds, named_diheds, unique_named_diheds):
    pots = []
    for frame in frames:
        coords = torch.transpose(frame, 0, 1)
        diheds = []
        for dihed in rel_diheds:
            coord1 = coords[rel_atoms_universe.index(dihed[0])]
            coord2 = coords[rel_atoms_universe.index(dihed[1])]
            coord3 = coords[rel_atoms_universe.index(dihed[2])]
            coord4 = coords[rel_atoms_universe.index(dihed[3])]
            vec1 = coord1 - coord2
            vec2 = coord3 - coord2
            vec3 = coord4 - coord3
            cross_product1 = torch.cross(vec1, vec2)
            cross_product2 = torch.cross(vec2, vec3)
            numerator = torch.dot(cross_product1, cross_product2)
            denominator = torch.norm(cross_product1) * torch.norm(cross_product2)
            dihedral_angle_radians = torch.atan2(numerator, denominator)
            diheds.append(dihedral_angle_radians)
        pot = 0
        for i in range(len(named_diheds)):
            ind = unique_named_diheds.index(named_diheds[i])
            v = v_vals[ind]
            n = n_vals[ind]
            gamma = gamma_vals[ind]
            if isinstance(v, list):
                for j in range(len(v)):
                    pot = pot + (v[j] * (1 + torch.cos((n[j] * diheds[i]) - (gamma[j] * (np.pi / 180)))))
            else:
                pot = pot + (v * (1 + torch.cos((n * diheds[i]) - (gamma * (np.pi / 180)))))
        pots.append(pot.cpu().detach())
    pots = np.array(pots)
    return pots

named_diheds_common = []
for dihed_common in rel_diheds_common:
    atom1 = rel_atoms_universe.index(dihed_common[0])
    atom2 = rel_atoms_universe.index(dihed_common[1])
    atom3 = rel_atoms_universe.index(dihed_common[2])
    atom4 = rel_atoms_universe.index(dihed_common[3])
    named_diheds_common.append(rel_atoms_universe_names[atom1] + ' - ' + rel_atoms_universe_names[atom2] \
                        + ' - ' + rel_atoms_universe_names[atom3] + ' - ' + rel_atoms_universe_names[atom4])
unique_named_diheds_common = list(set(named_diheds_common))
print(unique_named_diheds_common)
v_vals_common = [10.50]
n_vals_common = [2.0]
gamma_vals_common = [180.0]
divider_vals_common = [1.0]
def dihed_loss_common(frames, rel_atoms_universe, rel_diheds_common, named_diheds_common, unique_named_diheds_common):
    pots = []
    for frame in frames:
        coords = torch.transpose(frame, 0, 1)
        diheds_common = []
        for dihed_common in rel_diheds_common:
            coord1 = coords[rel_atoms_universe.index(dihed_common[0])]
            coord2 = coords[rel_atoms_universe.index(dihed_common[1])]
            coord3 = coords[rel_atoms_universe.index(dihed_common[2])]
            coord4 = coords[rel_atoms_universe.index(dihed_common[3])]
            vec1 = coord1 - coord2
            vec2 = coord3 - coord2
            vec3 = coord4 - coord3
            cross_product1 = torch.cross(vec1, vec2)
            cross_product2 = torch.cross(vec2, vec3)
            numerator = torch.dot(cross_product1, cross_product2)
            denominator = torch.norm(cross_product1) * torch.norm(cross_product2)
            dihedral_angle_radians = torch.atan2(numerator, denominator)
            diheds_common.append(dihedral_angle_radians)
        pot = 0
        for i in range(len(named_diheds_common)):
            ind = unique_named_diheds_common.index(named_diheds_common[i])
            v = v_vals_common[ind]
            n = n_vals_common[ind]
            gamma = gamma_vals_common[ind]
            if isinstance(v, list):
                for j in range(len(v)):
                    pot = pot + (v[j] * (1 + torch.cos((n[j] * diheds_common[i]) - (gamma[j] * (np.pi / 180)))))
            else:
                pot = pot + (v * (1 + torch.cos((n * diheds_common[i]) - (gamma * (np.pi / 180)))))
        pots.append(pot.cpu().detach())
    pots = np.array(pots)
    return pots

named_pairs = []
numbered_pairs = []
for pair in rel_pairs:
    atom1 = rel_atoms_universe.index(pair[0])
    atom2 = rel_atoms_universe.index(pair[1])
    named_pairs.append(rel_atoms_universe_names[atom1] + ' - ' + rel_atoms_universe_names[atom2])
    numbered_pairs.append((atom1, atom2))
unique_named_pairs = list(set(named_pairs))
print(unique_named_pairs)

R_min = {
    'N': '1.8240',
    'C': '1.9080',
    'CA': '1.9080',
    'CB': '1.9080',
    'O': '1.6612'
}
epsilon = {
    'N': '0.1700',
    'C': '0.0860',
    'CA': '0.0860',
    'CB': '0.0860',
    'O': '0.2100'
}
ep_ij_matrix = np.full((len(rel_atoms_universe_names), len(rel_atoms_universe_names)), -1, dtype=float)
R_ij_matrix = np.full((len(rel_atoms_universe_names), len(rel_atoms_universe_names)), -1, dtype=float)

for i in range(0,len(rel_atoms_universe_names)-1):
    for j in range(i+1,len(rel_atoms_universe_names)):
        ep_ij_matrix[i][j] = np.sqrt(float(epsilon[rel_atoms_universe_names[i]]) * float(epsilon[rel_atoms_universe_names[j]]))
for i in range(0,len(rel_atoms_universe_names)-1):
    for j in range(i+1,len(rel_atoms_universe_names)):
        R_ij_matrix[i][j] = (float(R_min[rel_atoms_universe_names[i]]) + float(R_min[rel_atoms_universe_names[j]])) / 2

A_ij_matrix = ep_ij_matrix * np.power(R_ij_matrix, 12)
B_ij_matrix = 2 * ep_ij_matrix * np.power(R_ij_matrix, 6)

def nonbonded_loss(frames, A_ij_matrix, B_ij_matrix, numbered_pairs):
    pots = []
    for frame in frames:
        coords = torch.transpose(frame, 0, 1)
        nums = coords.shape[0]
        coords_a = coords.unsqueeze(1).repeat(1, nums, 1)
        coords_b = coords.unsqueeze(0).repeat(nums, 1, 1)
        pairwise_distances = torch.sqrt(torch.sum((coords_a - coords_b)**2, dim=2))
        pairwise_distances = pairwise_distances.cpu().detach().numpy()
        mask = pairwise_distances > 10.0
        preserve_mask = np.zeros_like(pairwise_distances, dtype=bool)
        for index in numbered_pairs:
            preserve_mask[index] = True
        pairwise_distances[np.logical_and(mask, ~preserve_mask)] = -1.0
        #print(pairwise_distances.shape)
        positive_mask = pairwise_distances > 0.0
        first_term = np.zeros_like(A_ij_matrix, dtype=float)
        first_term[positive_mask] = A_ij_matrix[positive_mask] / np.power(pairwise_distances[positive_mask], 12)
        second_matrix = np.zeros_like(B_ij_matrix, dtype=float)
        second_matrix[positive_mask] = B_ij_matrix[positive_mask] / np.power(pairwise_distances[positive_mask], 6)
        LJ_pot = np.sum(first_term) + np.sum(second_matrix)
        pots.append(LJ_pot)
    pots = np.array(pots)
    return pots

class Custom_Loss(nn.Module):
    def __init__(self):
        super(Custom_Loss, self).__init__()
    
    def forward(self, interpolated, predicted, target, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds,\
               rel_angles, named_angles, unique_named_angles, rel_diheds, named_diheds, unique_named_diheds,\
               rel_diheds_common, named_diheds_common, unique_named_diheds_common, A_ij_matrix, B_ij_matrix, numbered_pairs):
        mse_losses = []
        amber_losses = []
        for i in range(target.shape[0]):
            out = predicted[i]
            inp = target[i]
            #print(out, inp)
            MSE_Loss = mseloss(out, inp)
            mse_losses.append(MSE_Loss)
        amber_loss = bonded_loss(interpolated, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds)\
                    + theta_loss(interpolated, rel_atoms_universe, rel_angles, named_angles, unique_named_angles) + \
                    + dihed_loss(interpolated, rel_atoms_universe, rel_diheds, named_diheds, unique_named_diheds) + \
                    + dihed_loss_common(interpolated, rel_atoms_universe, rel_diheds_common, named_diheds_common, unique_named_diheds_common) +\
                    + nonbonded_loss(interpolated, A_ij_matrix, B_ij_matrix, numbered_pairs)
        final_losses = []
        mse_losses = torch.stack(mse_losses).to('cuda')
        amber_loss = torch.tensor(amber_loss).to('cuda')
        with torch.no_grad():
            beta_parms = mse_losses / (10 * amber_loss)
        final_losses = mse_losses + (beta_parms * amber_loss)
#         for i in range(target.shape[0]):
#             with torch.no_grad():
#                 beta_parm = mse_losses[i] / (10 * amber_loss[i])
#             final_loss = mse_losses[i] + (beta_parm * amber_loss[i])
#             final_losses.append(final_loss)
#         final_losses = torch.tensor(final_losses)
        final_loss = torch.mean(final_losses)
        final_loss = torch.autograd.Variable(final_loss, requires_grad=True)
        return final_loss
    
class ResidualBlock(nn.Module):
    def __init__(self, f):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm1d(f),
                      nn.ReLU(inplace=True),
                      nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm1d(f)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
        # return torch.relu(x + self.conv_block(x))       #earlier runs were with 'return x + self.conv_block(x)' but not an issue (really?)


class To2D(nn.Module):

    def __init__(self):
        super(To2D, self).__init__()
        pass

    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2, 1))
        z = torch.sigmoid(z)
        return z


class From2D(nn.Module):
    def __init__(self):
        super(From2D, self).__init__()
        self.f = nn.Linear(2, 22*1)

    def forward(self, x):
        x = x.view(x.size(0), 2)
        #print(x.shape)
        x = self.f(x)
        #print(x.shape)
        x = x.view(x.size(0), 1, 22)
        #print(x.shape)
        return x


class Autoencoder(nn.Module):
    '''
    This is the autoencoder used in our `Ramaswamy 2021 paper <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052>`_.
    It is largely superseded by :func:`molearn.models.foldingnet.AutoEncoder`.
    '''
    def __init__(self, kernel=4, stride=2, padding=1, init_z=32, latent_z=1, depth=4, m=1.5, r=0, droprate=None):
        '''
        :param int init_z: number of channels in first layer
        :param int latent_z: number of latent space dimensions
        :param int depth: number of layers
        :param float m: scaling factor, dictating number of channels in subsequent layers
        :param int r: number of residual blocks between layers
        :param float droprate: dropout rate
        '''

        super(Autoencoder, self).__init__()
        # encoder block
        eb = nn.ModuleList()
        eb.append(nn.Conv1d(3, init_z, kernel, stride, padding, bias=False))
        eb.append(nn.BatchNorm1d(init_z))
        if droprate is not None:
            eb.append(nn.Dropout(p=droprate))
        eb.append(nn.ReLU(inplace=True))

        for i in range(depth):
            eb.append(nn.Conv1d(int(init_z*m**i), int(init_z*m**(i+1)), kernel, stride, padding, bias=False))
            eb.append(nn.BatchNorm1d(int(init_z*m**(i+1))))
            if droprate is not None:
                eb.append(nn.Dropout(p=droprate))
            eb.append(nn.ReLU(inplace=True))
            for j in range(r):
                eb.append(ResidualBlock(int(init_z*m**(i+1))))
        eb.append(nn.Conv1d(int(init_z*m**depth), latent_z, kernel, stride, padding, bias=False))
        eb.append(To2D())
        self.encoder = eb


        # decoder block
        db = nn.ModuleList()
        db.append(From2D())
        db.append(nn.ConvTranspose1d(latent_z, int(init_z*m**(depth)), kernel, stride, padding, bias=False))
        db.append(nn.BatchNorm1d(int(init_z*m**(depth))))
        if droprate is not None:
            db.append(nn.Dropout(p=droprate))
        db.append(nn.ReLU(inplace=True))
        for i in reversed(range(depth)):
            if int(init_z*m**i) == 72:
                db.append(nn.ConvTranspose1d(int(init_z*m**(i+1)), int(init_z*m**i), kernel, stride, padding, bias=False, output_padding=1))
            else:
                db.append(nn.ConvTranspose1d(int(init_z*m**(i+1)), int(init_z*m**i), kernel, stride, padding, bias=False))
            db.append(nn.BatchNorm1d(int(init_z*m**i)))
            if droprate is not None:
                db.append(nn.Dropout(p=droprate))
            db.append(nn.ReLU(inplace=True))
            for j in range(r):
                db.append(ResidualBlock(int(init_z*m**i)))
        db.append(nn.ConvTranspose1d(int(init_z*m**(i)), 3, kernel, stride, padding, output_padding=1))
        self.decoder = db

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
            #print(x.shape)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
            #print(x.shape)
        return x
    
# Create a sample input tensor (batch size, input channels, sequence length)
#input_tensor = torch.randn(15, 3, 100)

input_tensor = torch.randn(5, 3, 1417)

# Instantiate the model
model = Autoencoder()

# Encode
latent = model.encode(input_tensor)

# Print the latent layer shape
print("Latent shape:", latent.shape)

#Decode
output = model.decode(latent)

# Print the output shape
print("Output shape:", output.shape)

model = Autoencoder()
device = torch.device('cuda')
model = model.to(device)
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-5)

import math
training_start = time.time()
epochs = 400
outputs = []
losses = []
model.train()
epoch_losses = []
test_losses = []
nan_limit = 3
#encodings = []
for epoch in range(epochs):
    epoch_loss = []
    test_loss = []
    curr_nan = 0
    #test_enc = np.array([]).reshape(0,latent_dim)
    for i, image in enumerate(train_loader):
        print(i)
        sys.stdout.flush()
        optimizer.zero_grad()
        # print(image.dtype)
        # break4
        # image = image.to(torch.float32)
        image = image.to(device)
        latent = model.encode(image)
        latent = latent.to(device)
        num_outputs = latent.shape[0] // 2
        alpha = torch.rand(num_outputs, 2, 1)
        alpha = alpha.to(device)
        latent_interpolated = torch.zeros_like(latent)
        for i in range(num_outputs):
            latent_interpolated[i * 2] = (1 - alpha[i]) * latent[i * 2] + alpha[i] * latent[i * 2 + 1]
            latent_interpolated[i * 2 + 1] = (1 - alpha[i]) * latent[i * 2 + 1] + alpha[i] * latent[i * 2]
        reconstructed = model.decode(latent)
        interpolated = model.decode(latent_interpolated)
        #break
        criterion = Custom_Loss()
        loss = criterion(interpolated, reconstructed, image, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds,\
               rel_angles, named_angles, unique_named_angles, rel_diheds, named_diheds, unique_named_diheds,\
               rel_diheds_common, named_diheds_common, unique_named_diheds_common, A_ij_matrix, B_ij_matrix, numbered_pairs)
        #loss = torch.autograd.Variable(loss, requires_grad=True)
        
        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        losses.append(loss_item)
        epoch_loss.append(loss_item)
        #train_enc = train_enc + encoded_inp
        #outputs.append((epochs, image, reconstructed))
        if math.isnan(loss_item):
            if curr_nan == nan_limit:
                print(f"breaking because number of acceptable nan values exceeded at {i}")
                sys.stdout.flush()
                break
            print(f"nan found at i value of {i}")
            sys.stdout.flush()
            curr_nan = curr_nan + 1
        print(f"training epoch {epoch} : {i}/{len(train_loader)}, Loss: {loss_item}")
        sys.stdout.flush()
    curr_nan = 0
    for i, image in enumerate(test_loader):
        # print(image.dtype)
        # break4
        # image = image.to(torch.float32)
        image = image.to(device)
        latent = model.encode(image)
        latent = latent.to(device)
        num_outputs = latent.shape[0] // 2
        alpha = torch.rand(num_outputs, 2, 1)
        alpha = alpha.to(device)
        latent_interpolated = torch.zeros_like(latent)
        for i in range(num_outputs):
            latent_interpolated[i * 2] = (1 - alpha[i]) * latent[i * 2] + alpha[i] * latent[i * 2 + 1]
            latent_interpolated[i * 2 + 1] = (1 - alpha[i]) * latent[i * 2 + 1] + alpha[i] * latent[i * 2]
        reconstructed = model.decode(latent)
        interpolated = model.decode(latent_interpolated)
        criterion = Custom_Loss()
        loss = criterion(interpolated, reconstructed, image, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds,\
               rel_angles, named_angles, unique_named_angles, rel_diheds, named_diheds, unique_named_diheds,\
               rel_diheds_common, named_diheds_common, unique_named_diheds_common, A_ij_matrix, B_ij_matrix, numbered_pairs)
        #loss = torch.autograd.Variable(loss, requires_grad=True)
        loss_item = loss.item()
        test_loss.append(loss_item)
        if math.isnan(loss_item):
            if curr_nan == nan_limit:
                print(f"breaking because number of acceptable nan values exceeded at {i}")
                sys.stdout.flush()
                break
            print(f"nan found at i value of {i}")
            sys.stdout.flush()
            curr_nan = curr_nan + 1
        print(f"testing epoch {epoch} : {i}/{len(test_loader)}, Loss: {loss_item}")
        sys.stdout.flush()
    #if test_enc.size==0:
            #test_enc = np.vstack([test_enc,encoded_inp.cpu().detach().numpy()])
        #else:
            #test_enc = np.concatenate((test_enc,encoded_inp.cpu().detach().numpy()), axis=0)
        #print(encoded_inp.cpu().detach().numpy())
        #outputs.append((epochs, image, reconstructed))
        #print("epoch {} : {}/{}, Loss: {}".format(epoch, i, len(test_loader), loss.item()))
    epoch_losses.append(np.nanmean(epoch_loss))
    test_losses.append(np.nanmean(test_loss))

plt.plot(epoch_losses, label = "train")
plt.plot(test_losses, label = "test")
plt.legend()
plt.savefig(gro_file.split('.')[0]+"_LOSS.jpg")
plt.close()

total_train_time = time.time() - training_start

print(f"Total training time is {total_train_time}")
sys.stdout.flush()
losses_test= []
for i, image in enumerate(test_loader):
    image = image.to(device)
    latent = model.encode(image)
    latent = latent.to(device)
    num_outputs = latent.shape[0] // 2
    alpha = torch.rand(num_outputs, 2, 1)
    alpha = alpha.to(device)
    latent_interpolated = torch.zeros_like(latent)
    for i in range(num_outputs):
        latent_interpolated[i * 2] = (1 - alpha[i]) * latent[i * 2] + alpha[i] * latent[i * 2 + 1]
        latent_interpolated[i * 2 + 1] = (1 - alpha[i]) * latent[i * 2 + 1] + alpha[i] * latent[i * 2]
    reconstructed = model.decode(latent)
    interpolated = model.decode(latent_interpolated)
    criterion = Custom_Loss()
    loss = criterion(interpolated, reconstructed, image, rel_atoms_universe, rel_bonds, named_bonds, unique_named_bonds,\
           rel_angles, named_angles, unique_named_angles, rel_diheds, named_diheds, unique_named_diheds,\
           rel_diheds_common, named_diheds_common, unique_named_diheds_common, A_ij_matrix, B_ij_matrix, numbered_pairs)
    #loss = torch.autograd.Variable(loss, requires_grad=True)
    loss_item = loss.item()
    losses_test.append(loss_item)
    #outputs.append((epochs, image, reconstructed))
    print(f"Trained model testing: {i}/{len(test_loader)}, Loss: {loss_item}")
    sys.stdout.flush()
print("\n\nAVERAGE TEST LOSS WAS: " + str(np.nanmean(losses_test)) + "\n\n")
sys.stdout.flush()

torch.save(model,"modified_1DCNN.pth")
