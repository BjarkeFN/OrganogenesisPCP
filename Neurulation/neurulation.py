import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree
import scipy.io
import os
import itertools
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
if device == 'cuda':
    print('Using cuda')
    float_tensor = torch.cuda.FloatTensor
else:
    float_tensor = torch.FloatTensor


np.random.seed(1)


def init_alarulation(n,alpha0,pdiv0):
    P = scipy.io.loadmat('data/square1384wPCP.mat');
    P = P['p']
    x=P[:,0:3]
    p=P[:,3:6]
    q=P[:,6:9]
    print("qshape", q.shape)
    alpha=np.zeros((n,1));
    #x0=1;
    x0=3;
    alpha[abs(x[:,0])<x0]=alpha0;
    #alpha[:]=alpha0;
    pdiv=0*alpha;
    pdiv[abs(x[:,0])<x0]=pdiv0;

    return x, p, q, alpha, pdiv

def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf):
    tree = cKDTree(x)
    d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, n_jobs=-1)
    return d[:, 1:], idx[:, 1:]


def find_true_neighbours(d, dx):
    with torch.no_grad():
        z_masks = []
        i0 = 0
        batch_size = 100
        i1 = batch_size
        while True:
            if i0 >= dx.shape[0]:
                break

            n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
            n_dis += 1000 * torch.eye(n_dis.shape[1], device=device)[None, :, :]

            z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0  # check summatio dimension, etc.
            z_masks.append(z_mask)

            if i1 > dx.shape[0]:
                break
            i0 = i1
            i1 += batch_size
    z_mask = torch.cat(z_masks, dim=0)
    return z_mask


def potential(x, p, q, idx, d, lam, alpha, z_mask, dx,m ):

    # Calculate S
    # -> Introduce cell wedging
    anisotropy=1 # Toggle for anisotropic wedging
    pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
    pj = p[idx]
    qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
    qj = q[idx]
    aj = alpha[idx]
    ai = alpha[:,None,:].expand(p.shape[0], idx.shape[1],1)
    # Introduce \tilde{p}
    if anisotropy:
        qmean=(qi+qj)*0.5
        alphamean=(ai+aj)*0.5
        #qdotx = torch.sum(qmean * dx[:,idx,:], dim=2);
        qdotx = torch.sum(qmean * dx, dim=2);
        qdotx=qdotx[:,:,None].expand(p.shape[0],idx.shape[1],1)
        #print("Size qdotx: ", qdotx.shape)
        #print("Size alphamean: ", alphamean.shape)
        alphafactor=qdotx*alphamean;
        pti = pi-alphafactor*qmean
        ptj = pj+alphafactor*qmean
    else:
        alphamean=(ai+aj)*0.5
        pti = pi-alphamean*dx
        ptj = pj+alphamean*dx
    # Normalize ptilde
    pti = pti/torch.sqrt(torch.sum(pti ** 2, dim=2))[:, :, None]
    ptj = ptj/torch.sqrt(torch.sum(ptj ** 2, dim=2))[:, :, None]



    S1 = torch.sum(torch.cross(ptj, dx, dim=2) * torch.cross(pti, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    S = lam[0] * S1 + lam[1] * S2 + lam[2] * S3

    # Potential
    Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
    V = torch.sum(Vij)

    return V, int(m)


def init_simulation(dt, lam, p, q, x, alpha, pdiv):
    sqrt_dt = np.sqrt(dt)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=device)
    p = torch.tensor(p, requires_grad=True, dtype=torch.float, device=device)
    q = torch.tensor(q, requires_grad=True, dtype=torch.float, device=device)
    lam = torch.tensor(lam, dtype=torch.float, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    pdiv = torch.tensor(pdiv, dtype=torch.float, device=device)
    print(alpha.shape)
    return lam, p, q, sqrt_dt, x, alpha, pdiv


class TimeStepper:
    def __init__(self, init_k):
        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None

    def update_k(self, true_neighbour_max, tstep):
        k = self.k
        fraction = true_neighbour_max / k
        # Smart updating of k (number of cells to screen)
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return k, n_update

    def time_step(self, dt, eta, lam, p, q, pdiv, sqrt_dt, tstep, x, alpha, pdiv0):
        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        # Perform cell divisions
        n = x.shape[0]
        
        # Maximum number of cells in simulation:
        if n>2050:
            pdiv=0*pdiv;
            #pass
        randvec=np.random.rand(n, 1)
        # Keep track of cells which divide in this timestep:
        dividing=(randvec<pdiv.detach().to("cpu").numpy())
        numdiv=np.sum(dividing)
        if numdiv>0:
            with torch.no_grad():
                idxs = torch.tensor(np.where(dividing)[0], device=device)
                #idxs = torch.tensor([x.shape[0] - 1], device=device)
                dx_newcell=float_tensor(idxs.shape[0], 3).normal_();
                # Normalize dx_newcell to |dx|=2
                dx_newcell /= torch.sqrt(torch.sum(dx_newcell ** 2, dim=1))[:, None]
                # The distance to the mother cell at which a new cell is created:
                dx_newcell *= 1
                new_x = x[idxs].clone() + dx_newcell
                # Daughter cell inherits properties from mother cell:
                new_p = p[idxs].clone()
                new_q = q[idxs].clone()
                new_alpha = alpha[idxs].clone()
                new_pdiv = pdiv[idxs].clone()

            x = torch.cat((x.detach(), new_x), dim=0)
            x.requires_grad = True
            p = torch.cat((p.detach(), new_p), dim=0)
            p.requires_grad = True
            q = torch.cat((q.detach(), new_q), dim=0)
            q.requires_grad = True
            alpha = torch.cat((alpha.detach(), new_alpha), dim=0)
            alpha.requires_grad = True
            pdiv = torch.cat((pdiv.detach(), new_pdiv), dim=0)
            pdiv.requires_grad = True
            n = x.shape[0]

        assert q.shape == x.shape
        assert x.shape == p.shape

        k, n_update = self.update_k(self.true_neighbour_max, tstep)
        if tstep % n_update == 0 or self.idx is None or numdiv>0:
            d, idx = find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=device)
            self.d = torch.tensor(d, dtype=torch.float, device=device)
        idx = self.idx
        d = self.d


        # Normalise p, q
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]
        
        # Find true neighbours
        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        z_mask = find_true_neighbours(d, dx)
        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask, dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(idx, 1, sort_idx)
        m = torch.max(torch.sum(z_mask, dim=1)) + 1
        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]
        # Normalize dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]
        #print("zmaskshape", z_mask.shape)
        #print(z_mask)
        #print("idxshape", idx.shape)
        
        # Determine if a cell has alpha-neighbors
        epsi=1e-16;
        asum=torch.sum(alpha[idx]>0,dim=1)
        nonasum=torch.sum(alpha[idx]<epsi,dim=1)
        #pdiv0=max(pdiv);
        #pdiv0=3*0.002/36;
        pdiv[:,0]=pdiv0;
        pdiv=(asum>=2).float()*pdiv;
        pdiv=(nonasum>=3).float()*pdiv;
        pdiv=(alpha>0).float()*pdiv;
        
        # Calculate potential
        V, self.true_neighbour_max = potential(x, p, q, idx, d, lam, alpha, z_mask, dx, m)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            p += -p.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            q += -q.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt

        # Zero gradients
        x.grad.zero_()
        p.grad.zero_()
        q.grad.zero_()

        return x, p, q, alpha, pdiv


def simulation(x, p, q, alpha, pdiv, lam, pdiv0, eta, yield_every=1, dt=0.2):
    lam, p, q, sqrt_dt, x, alpha, pdiv = init_simulation(dt, lam, p, q, x, alpha, pdiv)
    #time_stepper = TimeStepper(init_k=100)
    time_stepper = TimeStepper(init_k=200)
    tstep = 0
    while True:
        tstep +=1
        x, p, q, alpha, pdiv = time_stepper.time_step(dt, eta, lam, p, q, pdiv, sqrt_dt, tstep, x, alpha, pdiv0)

        if tstep % yield_every == 0:
            xx = x.detach().to("cpu").numpy()
            pp = p.detach().to("cpu").numpy()
            qq = q.detach().to("cpu").numpy()
            aa = alpha.detach().to("cpu").numpy()
            ppdiv = pdiv.detach().to("cpu").numpy()
            yield xx, pp, qq, aa, ppdiv

        gc.collect()

# def simulation_plt(x, p, q, lam, eta, dt=0.1):
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     plt.ion()
#
#     lam, p, q, sqrt_dt, x = init_simulation(dt, lam, p, q, x)
#
#     k = 100
#     true_neighbour_max = k//2
#     for tstep in range(1000):
#         print(tstep)
#
#         x = time_step(dt, eta, k, lam, p, q, sqrt_dt, true_neighbour_max, tstep, x)
#
#         # Plot
#         if tstep % 1 == 0:
#             xx = x.detach().to("cpu").numpy()
#
#             ax.cla()
#             ax.scatter(xx[:, 0], xx[:, 1], xx[:, 2], c=xx[:, 1])
#             ax.set_xlim(-10, 10)
#             ax.set_ylim(-10, 10)
#             ax.set_zlim(-10, 10)
#             plt.draw()
#             plt.pause(.001)
#
#     print('Done')
#     plt.ioff()
#     plt.show()


def main(alpha0,pdiv0,k,l2,N):
    #n = 1599
    n = 1384
    x, p, q, alpha, pdiv = init_alarulation(n,alpha0,pdiv0)
    #lam = np.array([1.0, 0, 0])
    lam = np.array([1-l2, l2, 0])

    try:
        os.mkdir('data')
    except OSError:
        pass
    try:
        os.mkdir(f'mats_{k}')
    except OSError:
        pass


    data = []
    i = 0
    #N = 2000
    #x, p, q, alpha, pdiv, lam, eta, alpha0, l1, l3, r0, r1, yield_every=1, dt=0.2
    for xx, pp, qq, aa, ppdiv in itertools.islice(simulation(x, p, q, alpha, pdiv, lam, pdiv0, eta=0, yield_every=50), N):
        i += 1
        print(f'Running {i} of {N}', end='\r')
        data.append((xx,pp,qq,aa))
        scipy.io.savemat(f'mats_{k}/t{i}.mat', dict(x=xx,p=pp,q=qq,alpha=aa,pdiv=ppdiv))
    print(f'Simulation done, saved {N} datapoints')


if __name__ == '__main__':
    n=1384;
    # Parameters of the simulation:
    # alpha value for wedging:
    alpha0=0.5;
    # Basic division rate for dividing cells:
    pdiv0=2*0.002/36.0;
    # Number of datapoints to output:
    N=1400
    # k, a simple enumerator/identifier of the simulation
    k=1
    # l2, the strength of \lambda_2, the PCP coherence parameter.
    l2=9*0.05
    main(alpha0,pdiv0,k,l2,N)
