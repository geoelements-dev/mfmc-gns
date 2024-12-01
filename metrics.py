import torch 
import numpy as np

#Define distance loss metrics to calculate corrcoefs with
def chamfer_distance(points_a, points_b):
    #Inputs are Tensors that are still on the device, hence the following processing steps.
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # points_a_exp = points_a.unsqueeze(dim=1)
    # points_b_exp = points_b.unsqueeze(dim=0)
    # distances = torch.sum((points_a_exp - points_b_exp) ** 2, dim=-1)
    # min_dist_a_to_b = torch.min(distances, dim=1)[0]
    # min_dist_b_to_a = torch.min(distances, dim=0)[0]
    # chamfer_dist = torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)

    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.sum((a_exp - b_exp)**2, dim=-1)

        min_a_to_b = torch.mean(torch.min(distances, dim=1)[0])
        min_b_to_a = torch.mean(torch.min(distances, dim=0)[0])

        chamfer_dist = min_a_to_b + min_b_to_a
        elementwise_distances.append(chamfer_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Chamfer distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)

    # pairwise_distances = []
    # for a in points_a:
    #     row = []
    #     for b in points_b:
    #         dist = torch.sum((a-b)**2)
    #         row.append(dist)
    #     pairwise_distances.append(torch.stack(row))
    # dists = torch.stack(pairwise_distances)
    # min_dist_a_to_b = torch.min(dists, dim=1)[0]
    # min_dist_b_to_a = torch.min(dists, dim=0)[0]
    # chamfer_dist = torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)
    
    return chamfer_dist

def hausdorff_distance(points_a, points_b):
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # a_exp = points_a.unsqueeze(dim=1)
    # b_exp = points_b.unsqueeze(dim=0)
    # distances = torch.sum((a_exp - b_exp) **2, dim=-1)
    # haus_a_to_b = torch.max(torch.min(distances, dim=1)[0])
    # haus_b_to_a = torch.max(torch.min(distances, dim=0)[0])
    # hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)


    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.sum((a_exp - b_exp)**2, dim=-1)

        haus_a_to_b = torch.max(torch.min(distances, dim=1)[0])
        haus_b_to_a = torch.max(torch.min(distances, dim=0)[0])

        hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)
        elementwise_distances.append(hausdorff_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Hausdorff distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)

    # pairwise_distances = []
    # for a in points_a:
    #     row = []
    #     for b in points_b:
    #         dist = torch.sum((a-b)**2)
    #         row.append(dist)
    #     pairwise_distances.append(torch.stack(row))
    # dists = torch.stack(pairwise_distances)
    # haus_a_to_b = torch.max(torch.min(dists, dim=1)[0])
    # haus_b_to_a = torch.max(torch.min(dists, dim=0)[0])
    # hausdorff_dist = torch.max(haus_a_to_b, haus_b_to_a)

    # return hausdorff_dist

def wassterstein_metric(points_a, points_b, p=2):
    print("Input type: ", type(points_a))
    print("Input shape: ", len(points_a), points_a[0].shape)
    # points_a = torch.stack(points_a); points_b = torch.stack(points_b)
    # a_exp = points_a.unsqueeze(dim=1)
    # b_exp = points_b.unsqueeze(dim=0)
    # if p==2:
    #     distances = torch.sqrt(torch.sum((a_exp - b_exp)**p, dim=-1))
    # elif p==1:
    #     distances = torch.sum(torch.abs(a_exp - b_exp), dim=-1)
    # else:
    #     print(f"Error: Wassterstein distance not implemented for input p={p}. Valid choices are [1,2].")
    #     return

    elementwise_distances = []
    for a, b in zip(points_a, points_b):
        a_exp = a.unsqueeze(dim=1)
        b_exp = b.unsqueeze(dim=0)
        distances = torch.norm(a_exp - b_exp, p=p, dim=-1)

        wass_a_to_b = torch.mean(torch.min(distances, dim=1)[0])
        wass_b_to_a = torch.mean(torch.min(distances, dim=0)[0])

        wass_dist = torch.max(wass_a_to_b, wass_b_to_a)
        elementwise_distances.append(wass_dist)
    
    #ARBITRARY reduction step: we return the maximum distance between pairs here.
    print(f"Array of Wasserstein distances of len {len(elementwise_distances)}: {elementwise_distances}")
    return np.max(elementwise_distances)
    
    # distances = []
    # if p == 2:
    #     for a in points_a:
    #         row = []
    #         for b in points_b:
    #             dist = torch.sqrt(torch.sum((a-b)**p))
    #             row.append(dist)
    #         distances.append(torch.stack(row))
    # elif p == 1:
    #     for a in points_a:
    #         row = []
    #         for b in points_b:
    #             dist = torch.sum(torch.abs((a-b)))
    #             row.append(dist)
    #         distances.append(torch.stack(row))
    # distances = torch.stack(distances)
    
    # a_dist_sorted, a_idx = torch.sort(torch.min(distances, dim=1)[0])
    # b_dist_sorted, b_idx = torch.sort(torch.min(distances, dim=0)[0])
    # wasser_dist = torch.mean(torch.abs(a_dist_sorted - b_dist_sorted))

    # return wasser_dist

def center_of_mass(points):
    '''Takes in an array of points and returns the 2-norm of the center of mass.
    We assume equally weighted points for this calculation -- can add weighting func later.'''

    print("COM input array shape: ", np.shape(points))
    center = [np.mean(points[:,0]), np.mean(points[:,1])]
    print("Center", center)

    return np.sqrt(center[0]**2 + center[1]**2)

def get_velocities(points):
    '''Takes in a time evolution of points as a tensor (ntime, nparticles, 2).
    Returns the velocities calculated by finite difference, with the first set at zero.'''

    v = np.zeros(points.shape)
    dt = 1/points.shape[0]
    v[1:,:, :] = (points[1:,:,:] - points[:-1,:,:]) / dt

    return v

def delta_E(initial_points, final_points):
    '''Takes in two numpy arrays of points corresponding to initial and final positions.
    Returns the dissipation: taking delta_E as mgh_0 - mgh_f, this should be equal to KE plus dissipation.'''

    #This really returns h0 - hf, where h is the sum of the y positions of all points.

    #we ignore m and g here as they are constants.
    pe_0 = np.sum(initial_points[:,1], axis=-1)
    pe_f = np.sum(final_points[:,1], axis=-1)


    return pe_0 - pe_f

def kinetic_energy(positions_over_time):
    '''Input: a time evolution of particles.
    Output: kinetic energy of the particles, where for now we take mass to be unit.'''

    velos = get_velocities(positions_over_time)
    #Recall that the shape of positions_over_time will be (ntime, nparticles, 2).
    #Velocities is (ntime, nparticles, 2).
    #We want KE to be the sum of all particles and all time steps -- ie double sum over dims 0 and 1.

    normed_velos = np.linalg.norm(velos, axis=2) #norms the x and y to collapse to one value.
    ke = 0.5 * normed_velos ** 2
    total_ke = np.sum(ke)

    return total_ke
    
def dissipation(delta_e, ke):
    '''Returns the dissipation given total KE and total delE. Should be nonnegative!'''
    g = 9.81 #just using this for testing, need to determine how to extract from models.
    print(f"delE: {delta_e}")
    print(f"KE: {ke}")
    print(f"Calculated dissipation is {delta_e - ke/g}")
    return delta_e - ke/g