'''
'''
import numpy as np 

def build_param_file(i_mock, corr, scale='large', mock_catalog='Nseries'): 
    '''
    Build parameter files for CUTE
    '''
    
    if scale == 'large': 
        dim_max = 40.
        dim_nbin = 40
    elif scale == 'verysmall': 
        dim_max = 2.
        dim_nbin = 20
    else: 
        raise NotImplementedError

    if mock_catalog != 'Nseries': 
        raise ValueError
        
    # parameter file name
    param_file_name = ''.join([
        '/mount/riachuelo1/hahn/2pcf/param_files/param_',
        corr, 
        '_',
        mock_catalog,
        str(i_mock), 
        '_', 
        scale,
        '.txt'
        ])

    param_f = open(param_file_name, 'w')
    
    param_content = '\n'.join([
        "# input-output files and parameters", 
        "data_filename= /mount/riachuelo1/hahn/data/Nseries/CutskyN"+str(i_mock)+"."+corr+".cute2pcf.dat", 
        "random_filename= /mount/riachuelo1/hahn/data/Nseries/Nseries_cutsky_randoms_50x.2pcf.dat", 
        "input_format= 2", 
        "mask_filename= None", 
        "z_dist_filename= None", 
        "output_filename= /mount/riachuelo1/hahn/2pcf/corr/corr_2pcf_CutskyN"+str(i_mock)+"."+corr+".cute2pcf."+scale+"scale.dat", 
        "num_lines= all", 
        "", 
        "# estimation parameters", 
        "corr_type= 3D_ps", 
        "corr_estimator= LS", 
        "np_rand_fact= 8", 
        "", 
        "# cosmological parameters", 
        "omega_M= 0.31", 
        "omega_L= 0.69", 
        "w= -1", 
        "", 
        "# binning", 
        "log_bin= 0", 
        "n_logint= 10", 
        "dim1_max= "+str(round(dim_max, 2)), 
        "dim1_nbin= "+str(dim_nbin), 
        "dim2_max= "+str(round(dim_max, 2)), 
        "dim2_nbin= "+str(dim_nbin), 
        "dim3_min= 0.4", 
        "dim3_max= 0.7", 
        "dim3_nbin= 1", 
        "", 
        "# pixels for radial correlation", 
        "radial_aperture= 1", 
        "", 
        "# pm parameters", 
        "use_pm= 1", 
        "n_pix_sph= 2048"
        ])
    param_f.write(param_content)
    param_f.close()

    return param_file_name

def build_bao_param_file(i_mock, corr, mock_catalog='Nseries', f_down=None): 
    '''
    Build parameter files for CUTE
    '''
    
    dim_max = 150.
    dim_nbin = 30   # coarse binned

    if mock_catalog.lower() not in ['nseries', 'cmass']: 
        raise ValueError
    if f_down is None:
        raise ValueError
    elif (f_down != 1.0) and (mock_catalog.lower() == 'cmass'): 
        raise NotImplementedError
        
    # parameter file name
    if mock_catalog.lower() == 'nseries':
        param_file_name = ''.join([
            '/mount/riachuelo1/hahn/2pcf/param_files/param_',
            corr, 
            '_',
            mock_catalog,
            str(i_mock), 
            '_bao.txt'
            ])
    elif mock_catalog.lower() == 'cmass': 
        # CMASS does not have corrections or reaization #
        param_file_name = ''.join([
            '/mount/riachuelo1/hahn/2pcf/param_files/param_',
            mock_catalog,
            '_bao.txt'
            ])

    param_f = open(param_file_name, 'w')
    
    if mock_catalog.lower() == 'nseries': 
        param_content = '\n'.join([
            "# input-output files and parameters", 
            "data_filename= /mount/riachuelo1/hahn/data/Nseries/CutskyN"+str(i_mock)+"."+corr+".cute2pcf.dat", 
            "random_filename= /mount/riachuelo1/hahn/data/Nseries/Nseries_cutsky_randoms_50x_"+str(round(f_down,2))+"down.2pcf.dat", 
            "input_format= 2", 
            "mask_filename= None", 
            "z_dist_filename= None", 
            "output_filename= /mount/riachuelo1/hahn/2pcf/corr/corr_2pcf_CutskyN"+str(i_mock)+"."+corr+".cute2pcf."+str(round(f_down,2))+"down.BAOscale.dat", 
            "num_lines= all", 
            "", 
            "# estimation parameters", 
            "corr_type= 3D_ps", 
            "corr_estimator= LS", 
            "np_rand_fact= 8", 
            "", 
            "# cosmological parameters", 
            "omega_M= 0.31", 
            "omega_L= 0.69", 
            "w= -1", 
            "", 
            "# binning", 
            "log_bin= 0", 
            "n_logint= 10", 
            "dim1_max= "+str(round(dim_max, 2)), 
            "dim1_nbin= "+str(dim_nbin), 
            "dim2_max= "+str(round(dim_max, 2)), 
            "dim2_nbin= "+str(dim_nbin), 
            "dim3_min= 0.4", 
            "dim3_max= 0.7", 
            "dim3_nbin= 1", 
            "", 
            "# pixels for radial correlation", 
            "radial_aperture= 1", 
            "", 
            "# pm parameters", 
            "use_pm= 1", 
            "n_pix_sph= 2048"
            ])

    elif mock_catalog.lower() == 'cmass': 
        param_content = '\n'.join([
            "# input-output files and parameters", 
            "data_filename= /mount/riachuelo1/hahn/data/CMASS/cmass-dr12v4-N-Reid_fidcomso.cute2pcf.dat", 
            "random_filename= /mount/riachuelo1/hahn/data/CMASS/cmass-dr12v4-N-Reid_fidcomso.ran.cute2pcf.dat", 
            "input_format= 2", 
            "mask_filename= None", 
            "z_dist_filename= None", 
            "output_filename= /mount/riachuelo1/hahn/2pcf/corr/corr_2pcf_cmass-dr12v4-N-Reid.cute2pcf.BAOscale.dat", 
            "num_lines= all", 
            "", 
            "# estimation parameters", 
            "corr_type= 3D_ps", 
            "corr_estimator= LS", 
            "np_rand_fact= 8", 
            "", 
            "# cosmological parameters", 
            "omega_M= 0.31", 
            "omega_L= 0.69", 
            "w= -1", 
            "", 
            "# binning", 
            "log_bin= 0", 
            "n_logint= 10", 
            "dim1_max= "+str(round(dim_max, 2)), 
            "dim1_nbin= "+str(dim_nbin), 
            "dim2_max= "+str(round(dim_max, 2)), 
            "dim2_nbin= "+str(dim_nbin), 
            "dim3_min= 0.4", 
            "dim3_max= 0.7", 
            "dim3_nbin= 1", 
            "", 
            "# pixels for radial correlation", 
            "radial_aperture= 1", 
            "", 
            "# pm parameters", 
            "use_pm= 1", 
            "n_pix_sph= 2048"
            ])
    param_f.write(param_content)
    param_f.close()

    return param_file_name

def build_mockcatalog(i_mock, DorR='data', mock_catalog='Nseries'): 
    '''
    Preprocess mock catalogs so that it can be used with the CUTE package
    to calculate the two-point correlation function
    '''
    
    if mock_catalog.lower() == 'nseries': 
        data_dir = '/mount/riachuelo1/hahn/data/Nseries/'
    elif mock_catalog.lower() == 'cmass': 
        data_dir = '/mount/riachuelo1/hahn/data/CMASS/'

    if DorR == 'data': 
        if mock_catalog.lower() == 'nseries': 
            catalog_list = ['true', 'upweighted', 'collrm']
        elif mock_catalog.lower() == 'cmass': 
            catalog_list = ['true']

    elif DorR == 'random':
        catalog_list = ['random']

    for catalog in catalog_list: 

        if catalog == 'true': 
            if mock_catalog.lower() == 'nseries':
                file_name = ''.join([
                    data_dir, 
                    'CutskyN', str(i_mock), '.fidcosmo.dat'
                    ])
            elif mock_catalog.lower() == 'cmass': 
                file_name = ''.join([
                    data_dir, 
                    'cmass-dr12v4-N-Reid_fidcomso.dat'
                    ])
        elif catalog in ('upweighted', 'collrm'): 
            file_name = ''.join([
                data_dir,
                'CutskyN', str(i_mock), '.fidcosmo.fibcoll.dat'
                ])
        elif catalog in ('random'):
            if mock_catalog.lower() == 'nseries': 
                file_name = ''.join([
                    data_dir, 
                    'Nseries_cutsky_randoms_50x_redshifts_comp.dat'
                    ])
            elif mock_catalog.lower() == 'cmass': 
                file_name = ''.join([
                    data_dir, 
                    'cmass-dr12v4-N-Reid_fidcomso.ran.dat'
                    ])
        else: 
            raise NotImplementedError
    
        # import data 
        data = np.loadtxt(
                file_name, 
                unpack=True, 
                skiprows=1
                )
    
        # ra, dec, z, weight
        data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']
        
        if mock_catalog.lower() == 'nseries': 
            output_file = ''.join([
                data_dir, 
                'CutskyN', str(i_mock), '.', catalog, '.cute2pcf.dat'
                ])
        elif mock_catalog.lower() == 'cmass': 
            output_file = ''.join([
                data_dir, 
                'cmass-dr12v4-N-Reid_fidcomso.cute2pcf.dat'
                ])

        if catalog == 'true':           # True; no fiber collision
            
            if mock_catalog.lower() == 'nseries': 
                weight = 1.0 / data[4]
            elif mock_catalog.lower() == 'cmass': 
                weight = data[4] * (data[5] + data[6] - 1.0) / data[7]
                
            data_list = [data[0], data[1], data[2], weight]

        elif catalog == 'upweighted':   # upweighted

            rm_coll = np.where(data[3] > 0.) 
            weight = data[3] * (1.0 / data[4])
            data_list = [data[0][rm_coll], data[1][rm_coll], data[2][rm_coll], weight[rm_coll]]

        elif catalog == 'collrm':       # remove collided galaxy 

            rm_coll = np.where(data[3] > 0.)
            weight = (1.0 / data[4])
            data_list = [data[0][rm_coll], data[1][rm_coll], data[2][rm_coll], weight[rm_coll]]

        elif catalog == 'random': 
            
            if mock_catalog.lower() == 'nseries':
                weight = 1.0 / data[3]
            elif mock_catalog.lower() == 'cmass': 
                weight = 1.0 / data[4]

            data_list = [data[0], data[1], data[2], weight]
            
            if mock_catalog.lower() == 'nseries': 
                output_file = ''.join([
                    data_dir, 
                    'Nseries_cutsky_randoms_50x.2pcf.dat'
                    ])
            elif mock_catalog.lower() == 'cmass': 
                output_file = ''.join([
                    data_dir, 
                    'cmass-dr12v4-N-Reid_fidcomso.ran.cute2pcf.dat'
                    ])
        else: 
            raise NotImplementedError

        np.savetxt(
                output_file, 
                (np.vstack(np.array(data_list))).T, 
                delimiter='\t',
                fmt=data_fmts
                ) 

def build_downsampled_random(f_down):
    ''' 
    Downsample the ranodm catalog by a factor f_down

    N_ran,downsample = Nran * f_down 

    '''

    data_dir = '/mount/riachuelo1/hahn/data/Nseries/'
    file_name = ''.join([
        data_dir, 
        'Nseries_cutsky_randoms_50x_redshifts_comp.hdf5'
        ])

    f_rand = h5py.File(file_name, 'r')
    rand_data = f_rand['data'][:]

    Nran = len(rand_data[0])
    Nran_down = int(np.rint(f_down * np.float(Nran)))
    print 'Downsampled Nr = ', Nran_down
    
    # downsample randomly
    Nran_index = np.arange(Nran)
    np.random.seed()
    np.random.shuffle(Nran_index)
    keep_indices = Nran_index[:Nran_down]

    ra = rand_data[0][keep_indices]
    dec = rand_data[1][keep_indices]
    z = rand_data[2][keep_indices]
    comp = rand_data[3][keep_indices]
    weight = 1.0 / comp 

    
    data_fmts = ['%10.5f', '%10.5f', '%10.5f', '%10.5f']
    data_list = [ra, dec, z, weight]
    output_file = ''.join([
        data_dir, 
        "Nseries_cutsky_randoms_50x_", str(round(f_down,2)), "down.2pcf.dat", 
        ])
    print output_file 

    np.savetxt(
            output_file, 
            (np.vstack(np.array(data_list))).T, 
            delimiter='\t',
            fmt=data_fmts
            ) 

    return None

def cmass_2pcf_prop(): 
    #build_mockcatalog(1, DorR='data', mock_catalog='cmass')
    #build_mockcatalog(1, DorR='random', mock_catalog='cmass')
    print build_bao_param_file(1, 'true', mock_catalog='cmass', f_down=1.0)

    return None

if __name__=="__main__": 
    cmass_2pcf_prop()
    #build_downsampled_random(0.2)

    #for i_mock in xrange(1,85): 
    #    build_param_file(i_mock, 'true', scale='verysmall')
    #    build_param_file(i_mock, 'upweighted', scale='verysmall')
    #    print build_bao_param_file(i_mock, 'true', f_down=0.2)
    #    build_param_file(i_mock, 'collrm')
    #    #build_mockcatalog(i_mock, DorR='data')
