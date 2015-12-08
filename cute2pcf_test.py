import pickle
import numpy as np
import scipy as sp

def cute2pcf_tophat_amplitude(n_mocks, n_rp, n_pi, corrections=['true', 'upweighted'], scale='large'):

    for corr in corrections:    # for each correction
        for i_mock in n_mocks:  # for each mocks

            corr_file = ''.join([
                '/mount/riachuelo1/hahn/2pcf/corr/',
                'corr_2pcf_CutskyN', 
                str(i_mock), '.', 
                corr, 
                '.cute2pcf.', 
                scale, 
                'scale.dat'
                ])
            print corr_file 

            tpcf = np.loadtxt(corr_file, unpack=True)
    
            if i_mock == n_mocks[0]: 
                rp_bins = tpcf[0].reshape(n_rp, n_pi)[0]
                pi_bins = tpcf[1].reshape(n_rp, n_pi)[:,0]
                r_p, pi = np.meshgrid(rp_bins, pi_bins)

                twop_corr = tpcf[2].reshape(n_rp, n_pi)

            else: 
                twop_corr += tpcf[2].reshape(n_rp, n_pi)

        twop_corr /= np.float(len(n_mocks))  # average 2PCF
        
        print twop_corr.shape

        if corr == 'true': 
            true_corr = twop_corr.T
            continue
        
        twopcf_residual = 1.0 - (1.0 + twop_corr.T)/(1.0 + true_corr)
        rp_collided = np.where(np.abs(rp_bins) < 1.0)
        rp_notcollided = np.where(np.abs(rp_bins) > 0.4)
        print twopcf_residual[0]
        print twopcf_residual[:,0]
        print twopcf_residual[rp_collided[0], :]
        print twopcf_residual[rp_notcollided[0], :]

if __name__=="__main__": 
    cute2pcf_tophat_amplitude(range(1,2), 40, 40, corrections=['true', 'upweighted'], scale='large')

