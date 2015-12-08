import h5py 
import pickle
import numpy as np
import scipy as sp
import cosmolopy as cosmos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from halotools.mock_observables import clustering

# --- Local ---
from defutility.plotting import prettyplot
from defutility.plotting import prettycolors 

def catalog_2pcf(catalog): 
    """ Calculate the 2 Point Correlation Function for catalogs
    """
    
    data_dir = '/mount/riachuelo1/hahn/data/Nseries/'
    if catalog == 'true': 
        file_name = ''.join([
            data_dir, 
            'CutskyN1.fidcosmo.hdf5'
            ])
    elif catalog in ('fced', 'nocoll'): 
        file_name = ''.join([
            data_dir,
            'CutskyN1.fidcosmo.fibcoll.hdf5'
            ])

    data_sample, data_wfc = xyz_sample(file_name)

    rand_file = ''.join([
        data_dir, 
        'Nseries_cutsky_randoms_50x_redshifts_comp.hdf5'
        ])
    
    rand_sample, dump = xyz_sample(rand_file)

    N_r = len(rand_sample)
    rand_indices = range(N_r)
    np.random.seed()
    np.random.shuffle(rand_indices)
    N_r_reduced = int(np.rint(0.25 * np.float(N_r)))
    rand_sample = rand_sample[rand_indices[:N_r_reduced]]
    print 'Random downsampled from ', N_r, ' to ', N_r_reduced

    if catalog == 'fced':

        fced_indices = [] 
        for i_data, wfc in enumerate(data_wfc):

            fced_indices.append(i_data)

            while wfc > 1.0: 
                fced_indices.append(i_data)
                wfc -= 1.

        fced_indices = np.array(fced_indices)
        data_sample = data_sample[fced_indices]

    elif catalog == 'nocoll':
        data_sample = data_sample[np.where(data_wfc > 0.)]

    rp_bins = np.logspace(-1, 1.5, 31)
    pi_bins = np.linspace(0.0, 30.0, 31)
    
    # 2 Point Correlation Function
    twopcf = clustering.redshift_space_tpcf(
            data_sample, 
            rp_bins, 
            pi_bins, 
            estimator='Landy-Szalay', 
            N_threads='max', 
            randoms = rand_sample
            )
    
    # save 2pcf to pickel file 
    if catalog == 'true': 
        pickle_name = ''.join([
            data_dir, 
            '2pcf_CutskyN1.fidcosmo.p'
            ])
    elif catalog == 'fced': 
        pickle_name = ''.join([
            data_dir,
            '2pcf_CutskyN1.fidcosmo.fced.p'
            ])
    elif catalog == 'nocoll': 
        pickle_name = ''.join([
            data_dir,
            '2pcf_CutskyN1.fidcosmo.nocoll.p'
            ])

    pickle.dump([rp_bins, pi_bins, twopcf], open(pickle_name, "wb"))
        
    return None

def xyz_sample(file_name):
    """ 
    """
    omega_m = 0.31
    fidcosmo = {}
    fidcosmo['omega_M_0'] = omega_m
    fidcosmo['omega_lambda_0'] = 1.0 - omega_m 
    fidcosmo['h'] = 0.676
    fidcosmo = cosmos.distance.set_omega_k_0(fidcosmo)

    z_arr = np.arange(0.0, 1.01, 0.01)
    dm_arr = cosmos.distance.comoving_distance(z_arr, **fidcosmo) * fidcosmo['h']
    D_comov = sp.interpolate.interp1d(z_arr, dm_arr, kind='cubic') # cubic spline D_comov(z) for speed

    # import data 
    f_data = h5py.File(file_name, 'r') 
    data = f_data['data'][:]
    f_data.close()
   
    ra = data[0] * (np.pi/180.0)
    dec = data[1] * (np.pi/180.0)
    rad = D_comov(data[2])

    x = rad * np.sin(0.5*np.pi - dec) * np.cos(ra)
    y = rad * np.sin(0.5*np.pi - dec) * np.sin(ra)
    z = rad * np.cos(0.5*np.pi - dec)

    #print x[0], y[0], z[0]

    sample = np.array([x, y, z]).T
    
    # rotate x,y,z
    theta = np.pi/3.0
    rotation_matrix = np.array(
            [
                [np.cos(theta), 0., -np.sin(theta)], 
                [0., 1., 0.], 
                [np.sin(theta), 0., np.cos(theta)]
                ]
            )
    
    rot_sample = np.dot( sample, rotation_matrix ) 
    
    # xy plane
    xy_plane_circle = np.where(
            np.sqrt(rot_sample[:,0]**2 + rot_sample[:,1]**2) < 750.0 
            )

    final_sample = rot_sample[xy_plane_circle]
    wfc = data[3][xy_plane_circle]

    return final_sample, wfc

def cute_catalog_prep(i_mock): 
    '''
    '''
    data_dir = '/mount/riachuelo1/hahn/data/Nseries/'
    for catalog in ['true', 'fced', 'nocoll', 'rand']: 
        if catalog == 'true': 
            file_name = ''.join([
                data_dir, 
                'CutskyN', str(i_mock), '.fidcosmo.hdf5'
                ])
        elif catalog in ('fced', 'nocoll'): 
            file_name = ''.join([
                data_dir,
                'CutskyN', str(i_mock), '.fidcosmo.fibcoll.hdf5'
                ])
        elif catalog == 'rand': 
            file_name = ''.join([
                data_dir, 
                'Nseries_cutsky_randoms_50x_redshifts_comp.hdf5'
                ])
    
        # import data 
        f_data = h5py.File(file_name, 'r') 
        data = f_data['data'][:]
        f_data.close()

        if catalog == 'true': 
            data_list = [data[0], data[1], data[2], data[3]]
            output_file = ''.join([
                data_dir, 
                'CutskyN', str(i_mock), '.cute2pcf.dat'
                ])
            data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']
        elif catalog == 'fced': 
            # fiber collision upweight
            data_list = [data[0], data[1], data[2], data[3]]
            output_file = ''.join([
                data_dir, 
                'CutskyN', str(i_mock), '.fced.cute2pcf.dat'
                ])
            data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']
        elif catalog == 'nocoll':
            # no collided galaxy 
            nocoll = np.where(data[3] > 0.)
            wfc = np.array([1. for i in xrange(len(nocoll[0]))])
            
            data_list = [data[0][nocoll], data[1][nocoll], data[2][nocoll], wfc]
            output_file = ''.join([
                data_dir, 
                'CutskyN', str(i_mock), '.nocoll.cute2pcf.dat'
                ])
            data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']
        elif catalog == 'rand': 
            wfc = np.array([1. for i in xrange(len(data[0]))])
            data_list = [data[0], data[1], data[2], wfc]
            output_file = ''.join([
                data_dir, 
                'Nseries_cutsky_randoms_50x.2pcf.dat'
                ])
            data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']

        np.savetxt(
                output_file, 
                (np.vstack(np.array(data_list))).T, 
                delimiter='\t',
                fmt=data_fmts
                ) 

    return None

def plot_catalog_xyz(file_name):
    '''
    '''
    sample, wfc = xyz_sample(file_name)

    fig = plt.figure(figsize=(18,6))
    sub = fig.add_subplot(131)
    sub.scatter(sample[:,0], sample[:,1])
    sub.set_xlim([-1500.0, 1500.0])
    sub.set_ylim([-1500.0, 1500.0])
    sub.set_xlabel('X')
    sub.set_ylabel('Y')

    sub = fig.add_subplot(132)
    sub.scatter(sample[:,0], sample[:,2])
    sub.set_xlim([-1500.0, 1500.0])
    sub.set_ylim([0.0, 2000.0])
    sub.set_xlabel('X')
    sub.set_ylabel('z')
    
    sub = fig.add_subplot(133)
    sub.scatter(sample[:,1], sample[:,2])
    sub.set_xlim([-1500.0, 1500.0])
    sub.set_ylim([0.0, 2000.0])
    sub.set_xlabel('Y')
    sub.set_ylabel('z')

    plt.show()
    plt.close()
    return None

def plot_2pcf(catalog): 
    '''
    '''
    prettyplot()
    pretty_colors = prettycolors()

    data_dir = '/mount/riachuelo1/hahn/data/Nseries/'
    if catalog == 'true': 
        pickle_name = ''.join([
            data_dir, 
            '2pcf_CutskyN1.fidcosmo.p'
            ])
    elif catalog == 'fced': 
        pickle_name = ''.join([
            data_dir,
            '2pcf_CutskyN1.fidcosmo.fced.p'
            ])
    elif catalog == 'nocoll': 
        pickle_name = ''.join([
            data_dir,
            '2pcf_CutskyN1.fidcosmo.nocoll.p'
            ])

    pickle_data = pickle.load(open(pickle_name, "rb"))
    rp_bins = pickle_data[0]
    pi_bins = pickle_data[1]
    twopcf = pickle_data[2]
    #print twopcf

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)

    r_p, pi = np.meshgrid(rp_bins, pi_bins)
    sub.pcolormesh(r_p, pi, twopcf.T, cmap=plt.cm.afmhot)
    sub.set_xscale('log')
    sub.set_xlabel('$\mathtt{r_{p}}$', fontsize=40)
    sub.set_ylabel('$\pi$', fontsize=40)
    sub.set_xlim([0.0, 20.0])
    sub.set_ylim([0.0, 20.0])
    sub.set_title('N-series '+catalog.upper() + ' Catalog')
    plt.show()

    """
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    r_p, pi = np.meshgrid(rp_bins, pi_bins)
    sub.pcolormesh(r_p, pi, fced_2pcf.T, cmap=plt.cm.afmhot)
    sub.set_xscale('log')
    sub.set_xlabel('$r_{p}$', fontsize=40)
    sub.set_ylabel('$\pi$', fontsize=40)
    sub.set_xlim([0.0, 20.0])
    sub.set_ylim([0.0, 20.0])
    sub.set_title('Fiber Collided Catalog')
    
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    x = 0.5 * (rp_bins[:-1] + rp_bins[1:])
    y = 0.5 * (pi_bins[:-1] + pi_bins[1:])
    r_p, pi = np.meshgrid(x, y)
    #print np.log10(fced_2pcf.T/true_2pcf.T)
    #print fced_2pcf.T/true_2pcf.T
    implot = sub.imshow( true_2pcf.T-fced_2pcf.T, cmap=plt.cm.afmhot)
    implot.set_interpolation('none')
    #sub.pcolormesh(r_p, pi, true_2pcf.T-fced_2pcf.T, cmap=plt.cm.afmhot)
    #sub.contour(r_p, pi, fced_2pcf.T/true_2pcf.T-1.0, 20, cmap=plt.cm.afmhot)
    #sub.set_xscale('log')
    sub.set_xlabel('$r_{p}$', fontsize=40)
    sub.set_ylabel('$\pi$', fontsize=40)
    sub.set_xlim([0.0, 20.0])
    sub.set_ylim([0.0, 20.0])
    sub.set_title("True - Fibercollided")

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111)
    x = 0.5 * (rp_bins[:-1] + rp_bins[1:])
    y = 0.5 * (pi_bins[:-1] + pi_bins[1:])
    r_p, pi = np.meshgrid(x, y)
    #print np.log10(fced_2pcf.T/true_2pcf.T)
    #print fced_2pcf.T/true_2pcf.T
    sub.contourf(r_p, pi, fced_2pcf.T/true_2pcf.T-1.0, 
    """

if __name__=="__main__":
    cute_catalog_prep(1)
    #plot_2pcf('true')
    #plot_2pcf('fced')
    #plot_2pcf('nocoll')
    #print ''
    #catalog_2pcf('true')
    #print ''
    #catalog_2pcf('fced')
    #print ''
    #catalog_2pcf('nocoll')
