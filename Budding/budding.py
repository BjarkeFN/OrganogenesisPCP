import numpy as np
import numpy.matlib
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


def init_gastrumech(n):
    #x = np.load('data/square.npy')
    #p = 2 * np.random.rand(n, 3) - 1
    #q = 2 * np.random.randn(n, 3) - 1
    P = scipy.io.loadmat('data/square1384wPCP.mat');
    P = P['p']
    x=P[:,0:3]
    p=P[:,3:6]
    q=P[:,6:9]
    l1_0=0.5;
    lam = np.array([l1_0, 1-l1_0, 0])
    lam = np.matlib.repmat(lam,n,1)
    print("qshape", q.shape)
    alpha=np.zeros((n,1));
    r=np.sqrt(np.sum(x ** 2, 1))
    # Make "curl-around" PCP:
    rhat=x/r[:,None];
    q=np.cross(p,rhat);
    # Normalize q:
    q/=np.sqrt(np.sum(q ** 2, 1))[:, None]

    return x, p, q, alpha, lam

def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf):
    tree = cKDTree(x)
    d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, n_jobs=-1)
    return d[:, 1:], idx[:, 1:]


def find_true_neighbours(d, dx):
    with torch.no_grad():
        z_masks = []
        i0 = 0
        batch_size = 250
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
    anisotropy=0
    pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
    pj = p[idx]
    qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
    qj = q[idx]
    lami = lam[:, None, :].expand(p.shape[0], idx.shape[1], 3)
    lamj = lam[idx]
    # Rule for combining lambda_i and lambda_j:
    lam = (lami+lamj)*0.5;
    aj = alpha[idx]
    ai = alpha[:,None,:].expand(p.shape[0], idx.shape[1],1)
    # Introduce ptilde
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

    S = lam[:,:,0] * S1 + lam[:,:,1] * S2 + lam[:,:,2] * S3

    # Potential
    Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))
    V = torch.sum(Vij)

    return V, int(m)


def init_simulation(dt, lam, p, q, x, alpha):
    sqrt_dt = np.sqrt(dt)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=device)
    p = torch.tensor(p, requires_grad=True, dtype=torch.float, device=device)
    q = torch.tensor(q, requires_grad=True, dtype=torch.float, device=device)
    lam = torch.tensor(lam, dtype=torch.float, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    return lam, p, q, sqrt_dt, x, alpha


class TimeStepper:
    def __init__(self, init_k):
        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None

    def update_k(self, true_neighbour_max, tstep):
        k = self.k
        fraction = true_neighbour_max / k
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return k, n_update

    def time_step(self, dt, eta, lam, p, q, sqrt_dt, tstep, x, alpha, alpha0, l1, l3, r0, r1):
        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        #n = x.shape[0]

        assert q.shape == x.shape
        assert x.shape == p.shape
        
        # Update the "boundary conditions":
        rho=torch.sqrt(torch.sum(x[:,0:2] ** 2, 1))
        #print(r)
        with torch.no_grad():
            alpha[rho>r1]=0;
            alpha[rho<r1]=alpha0;
            alpha[rho<r0]=0;
            lam[:,0]=1;
            lam[:,1]=0;
            lam[:,2]=0;
            lam[rho<r1,0]=l1;
            lam[rho<r1,1]=1-(l1+l3);
            lam[rho<r1,2]=l3;
        alpha.requires_grad = True
        lam.requires_grad = True
        
        k, n_update = self.update_k(self.true_neighbour_max, tstep)
        if tstep % n_update == 0 or self.idx is None:
            d, idx = find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=device)
            self.d = torch.tensor(d, dtype=torch.float, device=device)
        idx = self.idx
        d = self.d
        
        k, n_update = self.update_k(self.true_neighbour_max, tstep)
        if tstep % n_update == 0 or self.idx is None:
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
        
        # Calculate potential
        V, self.true_neighbour_max = potential(x, p, q, idx, d, lam, alpha, z_mask, dx, m)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            x += -x.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            p += -p.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            # q is kept fixed
            #q += -q.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt

        # Zero gradients
        x.grad.zero_()
        p.grad.zero_()
        q.grad.zero_()

        return x, p, q, alpha, lam


def simulation(x, p, q, alpha, lam, alpha0, l1, l3, r0, r1, eta, yield_every=1, dt=0.2):
    lam, p, q, sqrt_dt, x, alpha = init_simulation(dt, lam, p, q, x, alpha)
    #time_stepper = TimeStepper(init_k=100)
    time_stepper = TimeStepper(init_k=200)
    tstep = 0
    while True:
        tstep +=1
        x, p, q, alpha, lam = time_stepper.time_step(dt, eta, lam, p, q, sqrt_dt, tstep, x, alpha, alpha0, l1, l3, r0, r1)

        if tstep % yield_every == 0:
            xx = x.detach().to("cpu").numpy()
            pp = p.detach().to("cpu").numpy()
            qq = q.detach().to("cpu").numpy()
            aa = alpha.detach().to("cpu").numpy()
            yield xx, pp, qq, aa

        gc.collect()

def main(alpha0,l1,l3,r0,r1,k):
    n = 1384
    x, p, q, alpha, lam = init_gastrumech(n)
    #lam = np.array([1.0, 0, 0])
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
    N = 2000
    for xx, pp, qq, aa, in itertools.islice(simulation(x, p, q, alpha, lam, alpha0, l1, l3, r0, r1, eta=0, yield_every=50), N):
        i += 1
        print(f'Running {i} of {N}', end='\r')
        data.append((xx,pp,qq,aa))
        scipy.io.savemat(f'mats_{k}/t{i}.mat', dict(x=xx,p=pp,q=qq,alpha=aa))
    print(f'Simulation done, saved {N} datapoints')


if __name__ == '__main__':
    alpha0=-0.5
    l1=0.5
    l3=0.1
    r0=5
    r1=10
    k=1
    main(alpha0,l1,l3,r0,r1,k)
