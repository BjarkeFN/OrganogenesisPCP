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
    pdiv=0*alpha;
    pdiv0=1e-4;
    #pdiv[r<r0]=0;


    return x, p, q, alpha, pdiv, lam

def loadsphere(n):
    #x = np.load('data/square.npy')
    #p = 2 * np.random.rand(n, 3) - 1
    #q = 2 * np.random.randn(n, 3) - 1
    np.random.seed(103834)
    #P = scipy.io.loadmat('data/sphere2000wPCP.mat');
    #P = scipy.io.loadmat('data/sphere2000wPCP_2TD_relax');
    #P = scipy.io.loadmat('data/sphere2000wPCP_dipole_relax.mat');
    P = scipy.io.loadmat('data/sphere4000');
    P = P['p']
    x=P[:,0:3]
    p=P[:,3:6]
    n = len(x)
    #q=P[:,6:9]

    n = len(x)
    # Center the structure:
    x = x-np.mean(x)
    # Initialize a curl-around PCP
    zhat1 = np.array([0, 0, 1])
    zhat = np.matlib.repmat(zhat1,n,1)
    q = -np.cross(p,zhat)
    # Normalize q:
    q/=np.sqrt(np.sum(q ** 2, 1))[:, None]

    alpha=np.zeros((n,1));
    print("qshape", q.shape)
    pdiv=0*alpha;
    l1_0=0.4;
    lam = np.array([l1_0, 1-l1_0, 0])
    lam = np.matlib.repmat(lam,n,1)
    topflag = x[:,2] > 15

    return x, p, q, alpha, pdiv, lam, topflag


def init_diskcreator(n):
    x = np.random.randn(n, 3)
    #x[:,2]=0
    p = 2 * np.random.rand(n, 3) - 1
    q = 2 * np.random.randn(n, 3) - 1
    p[:,0]=1
    p[:,1]=0
    p[:,2]=0

    return x, p, q

def init_random_system(n):
    x = np.random.randn(n, 3)
    p = 2 * np.random.rand(n, 3) - 1
    q = 2 * np.random.randn(n, 3) - 1

    return x, p, q


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
    #print("Size lam: ", lam.shape)
    #print("Size qj: ", qj.shape)
    #print("Size dx: ", dx.shape)
    aj = alpha[idx]
    ai = alpha[:,None,:].expand(p.shape[0], idx.shape[1],1)
    #print("Size ai: ", ai.shape)
    #print("Size aj: ", aj.shape)
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
    #pti=pi;
    #ptj=pj;
    # Normalize ptilde
    #pti /= torch.sqrt(torch.sum(pti ** 2, dim=1))[:, None]
    #ptj /= torch.sqrt(torch.sum(ptj ** 2, dim=1))[:, None]
    pti = pti/torch.sqrt(torch.sum(pti ** 2, dim=2))[:, :, None]
    ptj = ptj/torch.sqrt(torch.sum(ptj ** 2, dim=2))[:, :, None]


    #S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
    S1 = torch.sum(torch.cross(ptj, dx, dim=2) * torch.cross(pti, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    #S = z_mask.float() * (lam[0] * S1 + lam[1] * S2 + lam[2] * S3)
    S = lam[:,:,0] * S1 + lam[:,:,1] * S2 + lam[:,:,2] * S3
    # Add heaviside function
    #S = (S>0).float()*S;

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
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return k, n_update

    def time_step(self, dt, eta, lam, p, q, pdiv, sqrt_dt, tstep, x, alpha, alpha0, l1, l3, r0, r1, topflag):
        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive
        # Perform cell divisions
        n = x.shape[0]

        if n>0:
            pdiv=0*pdiv;
            #pass
        randvec=np.random.rand(n, 1)
        dividing=(randvec<pdiv.detach().to("cpu").numpy())
        #dividing=np.uint8(dividing)
        numdiv=np.sum(dividing)
        if numdiv>0:
            with torch.no_grad():
                idxs = torch.tensor(np.where(dividing)[0], device=device)
                #idxs = torch.tensor([x.shape[0] - 1], device=device)
                dx_newcell=float_tensor(idxs.shape[0], 3).normal_();
                # Normalize dx_newcell to |dx|=2
                dx_newcell /= torch.sqrt(torch.sum(dx_newcell ** 2, dim=1))[:, None]
                # Normally dx=2
                #dx_newcell *= 2
                # As an experiment, we let dx=1
                dx_newcell *= 1
                new_x = x[idxs].clone() + dx_newcell
                new_p = p[idxs].clone()
                new_q = q[idxs].clone()
                new_alpha = alpha[idxs].clone()
                new_lam = lam[idxs].clone()
                # NOTE: I have temporarily turned off "inheritance of division"
                # new_pdiv = 0*pdiv[idxs].clone()
                # Now it's turned on again:
                new_pdiv = pdiv[idxs].clone()

            x = torch.cat((x.detach(), new_x), dim=0)
            x.requires_grad = True
            p = torch.cat((p.detach(), new_p), dim=0)
            p.requires_grad = True
            q = torch.cat((q.detach(), new_q), dim=0)
            q.requires_grad = True
            lam = torch.cat((lam.detach(), new_lam), dim=0)
            lam.requires_grad = True
            alpha = torch.cat((alpha.detach(), new_alpha), dim=0)
            alpha.requires_grad = True
            pdiv = torch.cat((pdiv.detach(), new_pdiv), dim=0)
            pdiv.requires_grad = True
            n = x.shape[0]

        assert q.shape == x.shape
        assert x.shape == p.shape

        # Update the "boundary conditions":
        rho=torch.sqrt(torch.sum(x[:,0:2] ** 2, 1))
        #print(r)
        zcutoff = 1000
        z=x[:,2]
        a_bg = 0.0
        with torch.no_grad():
            alpha[rho>r1]=a_bg;
            alpha[rho<r1]=alpha0;
            alpha[rho<r0]=a_bg;
            alpha[z>zcutoff]=a_bg;
            lam[:,0]=1;
            lam[:,1]=0;
            lam[:,2]=0;
            lam[rho<r1,0]=l1;
            lam[rho<r1,1]=1-(l1+l3);
            lam[rho<r1,2]=l3;
            # Only a temporary thing ...
            lam[topflag,0]=1;
            lam[topflag,1]=0;
            lam[topflag,2]=0;
            alpha[topflag]=a_bg;
        alpha.requires_grad = True
        lam.requires_grad = True

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
        pdiv[:,0]=3*0.002/36.0;
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
            # q is kept fixed
            #q += -q.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt

        # Zero gradients
        x.grad.zero_()
        p.grad.zero_()
        q.grad.zero_()

        return x, p, q, alpha, pdiv, lam


def simulation(x, p, q, alpha, pdiv, lam, alpha0, l1, l3, r0, r1, topflag, eta, yield_every=1, dt=0.2):
    lam, p, q, sqrt_dt, x, alpha, pdiv = init_simulation(dt, lam, p, q, x, alpha, pdiv)
    topflag = x[:,2] > 15
    #time_stepper = TimeStepper(init_k=100)
    time_stepper = TimeStepper(init_k=200)
    tstep = 0
    while True:
        tstep +=1
        x, p, q, alpha, pdiv, lam = time_stepper.time_step(dt, eta, lam, p, q, pdiv, sqrt_dt, tstep, x, alpha, alpha0, l1, l3, r0, r1, topflag)

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


def main(alpha0,l1,l3,r0,r1,k):
    #n = 1599
    n = 2000
    x, p, q, alpha, pdiv, lam, topflag = loadsphere(n)
    #lam = np.array([1.0, 0, 0])
    try:
        os.mkdir('data')
    except OSError:
        pass
    try:
        os.mkdir(f'mats_{k}')
    except OSError:
        pass


    xdata = []
    pdata = []
    qdata = []
    adata = []
    pdivdata = []
    xx=x
    pp=p
    qq=q
    aa=alpha
    ppdiv=pdiv
    i = 0
    xdata.append(xx)
    pdata.append(pp)
    qdata.append(qq)
    adata.append(aa)
    pdivdata.append(ppdiv)
    #N = 700 # (with dt = 0.1)
    N = 5000
    #x, p, q, alpha, pdiv, lam, eta, alpha0, l1, l3, r0, r1, yield_every=1, dt=0.2
    noiselevel = 5e-2
    for xx, pp, qq, aa, ppdiv in itertools.islice(simulation(x, p, q, alpha, pdiv, lam, alpha0, l1, l3, r0, r1, topflag, eta=noiselevel, yield_every=50, dt=0.1), N):
        i += 1
        print(f'Running {i} of {N}', end='\r')
        xdata.append(xx)
        pdata.append(pp)
        qdata.append(qq)
        adata.append(aa)
        pdivdata.append(ppdiv)
        if i % 1 == 0:
            scipy.io.savemat(f'mats_{k}/t{i}.mat', dict(x=xx,p=pp,q=qq,alpha=aa,pdiv=ppdiv))
        if i % 100 == 0:
            xOUTdata = np.array(xdata)
            pOUTdata = np.array(pdata)
            qOUTdata = np.array(qdata)
            aOUTdata = np.array(adata)
            pdivOUTdata = np.array(pdivdata)
            np.save(f'data/xdata_{n}.npy', xOUTdata)
            np.save(f'data/pdata_{n}.npy', pOUTdata)
            np.save(f'data/qdata_{n}.npy', qOUTdata)
            np.save(f'data/adata_{n}.npy', aOUTdata)
            np.save(f'data/pdivdata_{n}.npy', pdivOUTdata)
            print(f'Saved {N} datapoints')
    print(f'Simulation done.')


if __name__ == '__main__':
    # This is how it was originally
    #alpha0=-0.5
    #l1=0.5
    #l3=0.1
    alpha0=0.4
    l1=0.5
    l3=0.1
    r0=7
    r1=21
    # Starting from k= ...
    k=1
    main(alpha0,l1,l3,r0,r1,k)
