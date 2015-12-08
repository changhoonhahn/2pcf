'''

Plotting module for CUTE generated Two Point Correlation Functions xi(rp, pi)

'''
import numpy as np
import scipy as sp
import cosmolopy as cosmos
import matplotlib.pyplot as plt

# -- Local --
from defutility.plotting import prettyplot
from defutility.plotting import prettycolors 

def plot_cute2pcf(n_mocks, n_rp, n_pi, corrections=['true', 'upweighted', 'collrm'], scale='large', **kwargs): 
    '''
    Plot xi(r_p, pi) from CUTE 2PCF code
    '''

    prettyplot()
    pretty_colors = prettycolors()

    if scale == 'large': 
        contour_range = np.arange(-3.0, 2.5, 0.5)
    elif scale == 'small': 
        contour_range = np.arange(-0.6, 3.0, 0.15)
    elif file_flag == 'smaller': 
        contour_range = np.arange(-0.4, 3.2, 0.2)

    for corr in corrections:    # for each correction

        fig = plt.figure(figsize=(15,10))
        sub = fig.add_subplot(111)

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
        
        # contour of log(xi(r_p, pi))
        cont = sub.contourf(r_p, pi, np.log10(twop_corr.T), contour_range, cmap=plt.cm.afmhot)
        #sub.contour(r_p, pi, np.log10(twop_corr.T), contour_range, linewidths=4, cmap=plt.cm.afmhot)
        plt.colorbar(cont)

        sub.vlines(0.4, 0.0, np.max(r_p), lw=4, linestyle='--', color='red')

        sub.set_xlabel('$\mathtt{r_{p}}$', fontsize=40)
        sub.set_ylabel('$\pi$', fontsize=40)
        sub.set_xlim([np.min(rp_bins), np.max(rp_bins)])
        sub.set_ylim([np.min(pi_bins), np.max(pi_bins)])
         
        sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;log\xi(r_{||}, r_\perp)$", fontsize=40)
    
        fig_name = ''.join([
            '/home/users/hahn/powercode/FiberCollisions/figure/',
            '2pcf_Nseries_', corr, '_', str(len(n_mocks)), 'mocks.', scale, '.png'
            ])

        fig.savefig(fig_name, bbox_inches="tight")
        plt.close()

def plot_bao_cute2pcf(n_mocks, n_rp, n_pi, corrections=['true'], f_down=0.2, xiform='asinh', cmap='hot', **kwargs): 
    '''
    Plot xi(r_p, pi) from CUTE 2PCF code
    '''

    prettyplot()
    mpl.rcParams['font.size']=24
    pretty_colors = prettycolors()

    contour_range = 10 

    for corr in corrections:    # for each correction

        fig = plt.figure(figsize=(14,10))
        sub = fig.add_subplot(111)

        for i_mock in n_mocks:  # for each mocks

            corr_file = ''.join([
                '/mount/riachuelo1/hahn/2pcf/corr/',
                'corr_2pcf_CutskyN', 
                str(i_mock), '.', 
                corr, 
                '.cute2pcf.', str(round(f_down,2)), 'down.BAOscale.dat'
                ])
            print corr_file 

            tpcf = np.loadtxt(corr_file, unpack=True)
    
            if i_mock == n_mocks[0]: 

                rp_bins = tpcf[0].reshape(n_rp, n_pi)[0]
                pi_bins = tpcf[1].reshape(n_rp, n_pi)[:,0]

                twop_corr = tpcf[2].reshape(n_rp, n_pi)
            
            else: 
                twop_corr += tpcf[2].reshape(n_rp, n_pi)

        twop_corr /= np.float(len(n_mocks))  # average 2PCF
                
        quad_rp_bins = np.hstack([-1.0 * rp_bins[::-1], rp_bins])
        quad_pi_bins = np.hstack([-1.0 * pi_bins[::-1], pi_bins])
        
        quad_tpcf_0, quad_tpcf_1, quad_tpcf_2 = [], [], []
        for i_pi, pibin in enumerate(quad_pi_bins): 
            for i_rp, rpbin in enumerate(quad_rp_bins): 
                quad_tpcf_0.append(rpbin)
                quad_tpcf_1.append(pibin)
                if i_pi < n_pi: 
                    ipi = n_pi - i_pi - 1
                else: 
                    ipi = i_pi % n_pi

                if i_rp < n_rp: 
                    irp = n_rp - i_rp - 1
                else: 
                    irp = i_rp % n_rp
                quad_tpcf_2.append(twop_corr[irp, ipi])
                
                #if (np.abs(rpbin) < 10.) and (np.abs(pibin) >95.): 
                #    print twop_corr[irp, ipi]

        quad_tpcf_0 = np.array(quad_tpcf_0)
        quad_tpcf_1 = np.array(quad_tpcf_1)
        quad_tpcf_2 = np.array(quad_tpcf_2)
        
        quad_rp_bins = quad_tpcf_0.reshape(2*n_rp, 2*n_pi)[0]
        quad_pi_bins = quad_tpcf_1.reshape(2*n_rp, 2*n_pi)[:,0]
        quad_twop_corr = quad_tpcf_2.reshape(2*n_rp, 2*n_pi)

        quad_rp, quad_pi = np.meshgrid(quad_rp_bins, quad_pi_bins)

        if cmap == 'hot':
            colormap = plt.cm.afmhot
        else: 
            colormap = plt.cm.Paired
        
        # contour of log(xi(r_p, pi))
        if xiform == 'log': 
            print np.min(quad_twop_corr), np.max(quad_twop_corr)
            norm = mpl.colors.SymLogNorm(0.001, vmin=-0.01, vmax=5.0, clip=True)
            cont = sub.pcolormesh(quad_rp, quad_pi, quad_twop_corr, norm=norm, cmap=colormap)
            #cont = sub.contourf(quad_rp, quad_pi, np.log10(quad_twop_corr), contour_range, cmap=plt.cm.afmhot)

            ticker = mpl.ticker.FixedLocator([-0.01, 0.0, 0.5, 0.1, 1.0, 3.0])

        elif xiform == 'asinh': 
            contour_range = np.arange(-0.025, 0.105, 0.005)
            #cont = sub.contourf(quad_rp, quad_pi, np.arcsinh(10. * quad_twop_corr), contour_range, cmap=plt.cm.afmhot)
            norm = mpl.colors.Normalize(-0.1, 0.1, clip=True)
            cont = sub.pcolormesh(quad_rp, quad_pi, np.arcsinh(10. * quad_twop_corr), norm=norm, cmap=colormap)
            #cont.set_interpolation('none')

        elif xiform == 'none': 
            contour_range = np.arange(-0.5, 5.0, 0.05)
            cont = sub.contourf(quad_rp, quad_pi, quad_twop_corr, contour_range, cmap=colormap)
            #cont = sub.contourf(r_p, pi, twop_corr.T, contour_range, cmap=plt.cm.afmhot)
        else:
            raise ValueError

        #sub.contour(r_p, pi, np.log10(twop_corr.T), contour_range, linewidths=4, cmap=plt.cm.afmhot)
        if xiform == 'log':
            plt.colorbar(cont, ticks=ticker)
        else: 
            plt.colorbar(cont)

        sub.set_xlabel('$\mathtt{r_{\perp}} \; (\mathtt{Mpc}/h)$', fontsize=50)
        sub.set_ylabel('$\mathtt{r_{||}} \; (\mathtt{Mpc}/h)$', fontsize=50)

        sub.set_xlim([np.min(quad_rp_bins), np.max(quad_rp_bins)])
        sub.set_ylim([np.min(quad_pi_bins), np.max(quad_pi_bins)])

        if cmap == 'hot': 
            colormap_str = 'hot'
        else: 
            colormap_str = 'jet'

        
        if xiform == 'log': 
            #sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;log\xi(r_{||}, r_\perp)$", fontsize=40)
            fig_name = ''.join([
                '/home/users/hahn/powercode/FiberCollisions/figure/',
                'log2pcf_Nseries_', corr, '_', str(len(n_mocks)), 'mocks.', str(round(f_down,2)), 'down.BAO.', colormap_str ,'.png'
                ])
        elif xiform == 'asinh': 
            sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;arcsinh\:10\times\xi(r_{||}, r_\perp)$", fontsize=40)
            fig_name = ''.join([
                '/home/users/hahn/powercode/FiberCollisions/figure/',
                'arcsinh2pcf_Nseries_', corr, '_', str(len(n_mocks)), 'mocks.', str(round(f_down,2)), 'down.BAO.', colormap_str, '.png'
                ])
        elif xiform == 'none': 
            sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;\xi(r_{||}, r_\perp)$", fontsize=40)
            fig_name = ''.join([
                '/home/users/hahn/powercode/FiberCollisions/figure/',
                '2pcf_Nseries_', corr, '_', str(len(n_mocks)), 'mocks.', str(round(f_down,2)), 'down.BAO.', colormap_str, '.png'
                ])
    
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        #fig.savefig(fig_name, bbox_inches="tight")
        #plt.close()

def plot_bao_cmass_cute2pcf(n_rp, n_pi, xiform='asinh', cmap='hot', **kwargs): 
    '''
    Plot xi(r_p, pi) from CUTE 2PCF code specifically for CMASS
    '''

    prettyplot()
    mpl.rcParams['font.size']=24
    pretty_colors = prettycolors()

    contour_range = 10 

    fig = plt.figure(figsize=(14,10))
    sub = fig.add_subplot(111)

    corr_file = ''.join([
        '/mount/riachuelo1/hahn/2pcf/corr/',
        'corr_2pcf_cmass-dr12v4-N-Reid.cute2pcf.BAOscale.dat'
        ])
    print corr_file 

    tpcf = np.loadtxt(corr_file, unpack=True)
    
    rp_bins = tpcf[0].reshape(n_rp, n_pi)[0]
    pi_bins = tpcf[1].reshape(n_rp, n_pi)[:,0]

    twop_corr = tpcf[2].reshape(n_rp, n_pi)
    
    # annoying stuff to reflect the quadrants
    quad_rp_bins = np.hstack([-1.0 * rp_bins[::-1], rp_bins])
    quad_pi_bins = np.hstack([-1.0 * pi_bins[::-1], pi_bins])
        
    quad_tpcf_0, quad_tpcf_1, quad_tpcf_2 = [], [], []
    for i_pi, pibin in enumerate(quad_pi_bins): 
        for i_rp, rpbin in enumerate(quad_rp_bins): 
            quad_tpcf_0.append(rpbin)
            quad_tpcf_1.append(pibin)
            if i_pi < n_pi: 
                ipi = n_pi - i_pi - 1
            else: 
                ipi = i_pi % n_pi

            if i_rp < n_rp: 
                irp = n_rp - i_rp - 1
            else: 
                irp = i_rp % n_rp
            quad_tpcf_2.append(twop_corr[irp, ipi])
            
            #if (np.abs(rpbin) < 10.) and (np.abs(pibin) >95.): 
            #    print twop_corr[irp, ipi]

    quad_tpcf_0 = np.array(quad_tpcf_0)
    quad_tpcf_1 = np.array(quad_tpcf_1)
    quad_tpcf_2 = np.array(quad_tpcf_2)
        
    quad_rp_bins = quad_tpcf_0.reshape(2*n_rp, 2*n_pi)[0]
    quad_pi_bins = quad_tpcf_1.reshape(2*n_rp, 2*n_pi)[:,0]
    quad_twop_corr = quad_tpcf_2.reshape(2*n_rp, 2*n_pi)

    quad_rp, quad_pi = np.meshgrid(quad_rp_bins, quad_pi_bins)

    if cmap == 'hot':
        colormap = plt.cm.afmhot
    else: 
        colormap = plt.cm.Paired
    
    # contour of log(xi(r_p, pi))
    if xiform == 'log': 
        print np.min(quad_twop_corr), np.max(quad_twop_corr)
        norm = mpl.colors.SymLogNorm(0.005, vmin=-0.02, vmax=4.0, clip=True)
        cont = sub.pcolormesh(quad_rp, quad_pi, quad_twop_corr, norm=norm, cmap=colormap)

        ticker = mpl.ticker.FixedLocator([-0.01, 0.0, 0.5, 0.1, 1.0, 3.0])

    elif xiform == 'asinh': 
        contour_range = np.arange(-0.025, 0.105, 0.005)
        #cont = sub.contourf(quad_rp, quad_pi, np.arcsinh(10. * quad_twop_corr), contour_range, cmap=plt.cm.afmhot)
        norm = mpl.colors.Normalize(-0.1, 0.1, clip=True)
        cont = sub.pcolormesh(quad_rp, quad_pi, np.arcsinh(10. * quad_twop_corr), norm=norm, cmap=colormap)
        #cont.set_interpolation('none')

    elif xiform == 'none': 
        contour_range = np.arange(-0.5, 5.0, 0.05)
        cont = sub.contourf(quad_rp, quad_pi, quad_twop_corr, contour_range, cmap=colormap)
        #cont = sub.contourf(r_p, pi, twop_corr.T, contour_range, cmap=plt.cm.afmhot)
    else:
        raise ValueError

    if xiform == 'log':
        plt.colorbar(cont, ticks=ticker)
    else: 
        plt.colorbar(cont)

    sub.set_xlabel('$\mathtt{r_{\perp}} \; (\mathtt{Mpc}/h)$', fontsize=50)
    sub.set_ylabel('$\mathtt{r_{||}} \; (\mathtt{Mpc}/h)$', fontsize=50)

    sub.set_xlim([np.min(quad_rp_bins), np.max(quad_rp_bins)])
    sub.set_ylim([np.min(quad_pi_bins), np.max(quad_pi_bins)])

    if cmap == 'hot': 
        colormap_str = 'hot'
    else: 
        colormap_str = 'jet'

    
    if xiform == 'log': 
        fig_name = ''.join([
            '/home/users/hahn/powercode/FiberCollisions/figure/',
            'log2pcf_cmass-dr12v4-N-Reid.BAO.', colormap_str ,'.png'
            ])
    elif xiform == 'asinh': 
        sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;arcsinh\:10\times\xi(r_{||}, r_\perp)$", fontsize=40)
        fig_name = ''.join([
            '/home/users/hahn/powercode/FiberCollisions/figure/',
            'arcsinh2pcf_cmass-dr12v4-N-Reid.BAO.', colormap_str ,'.png'
            ])
    elif xiform == 'none': 
        sub.set_title(r'$\mathtt{'+corr.upper()+r"}\;\xi(r_{||}, r_\perp)$", fontsize=40)
        fig_name = ''.join([
            '/home/users/hahn/powercode/FiberCollisions/figure/',
            '2pcf_cmass-dr12v4-N-Reid.BAO.', colormap_str ,'.png'
            ])

    plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(fig_name, bbox_inches="tight")
    plt.close()

def plot_cute2pcf_residual(n_mocks, n_rp, n_pi, corrections=['true', 'upweighted', 'collrm'], scale='large', **kwargs): 
    '''
    Plot xi(r_p, pi) from CUTE 2PCF code
    '''

    prettyplot()
    pretty_colors = prettycolors()

    if scale == 'large': 
        contour_range = np.arange(-0.05, 0.05, 0.005)
    elif scale == 'small': 
        contour_range = np.arange(-0.1, 0.11, 0.01)
    elif scale == 'smaller': 
        contour_range = np.arange(-0.5, 0.5, 0.05)
    elif scale == 'verysmall': 
        contour_range = 20

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

        if corr == 'true': 
            true_corr = twop_corr.T
            continue
        print (twop_corr.T).shape

        fig = plt.figure(figsize=(15,10))
        sub = fig.add_subplot(111)

        # contour of 1 - (1 + xi^fc)/(1+xi^true)
        residual_2pcf = 1.0 - (1.0 + twop_corr.T)/(1.0 + true_corr)
        print np.max(residual_2pcf)
        cont = sub.contourf(
                r_p, pi, 
                residual_2pcf, 
                contour_range, 
                cmap=plt.cm.afmhot
                )
        plt.colorbar(cont)

        sub.vlines(0.4, 0.0, np.max(r_p), lw=4, linestyle='--', color='red')

        sub.set_xlabel('$\mathtt{r_{p}}$', fontsize=40)
        sub.set_ylabel('$\pi$', fontsize=40)
        sub.set_xlim([np.min(rp_bins), np.max(rp_bins)])
        sub.set_ylim([np.min(pi_bins), np.max(pi_bins)])
        
        sub.set_title(r"$1 - (1 + \xi^\mathtt{"+corr.upper()+r"})/(1+ \xi^\mathtt{TRUE})$", fontsize=40)
    
        fig_name = ''.join([
            '/home/users/hahn/powercode/FiberCollisions/figure/',
            '2pcf_Nseries_', corr, '_', str(len(n_mocks)), 'mocks.', scale, '.tophat_comparison.png'
            ])

        fig.savefig(fig_name, bbox_inches="tight")
        plt.close()

if __name__=="__main__":
    #plot_cute2pcf_residual(range(1,21), 20, 20, corrections=['true', 'upweighted'], scale='verysmall')
    plot_bao_cmass_cute2pcf(30, 30, xiform = 'log', cmap='hot')
    #plot_bao_cute2pcf(range(1,45), 30, 30, xiform = 'none', corrections=['true'])
    #plot_bao_cute2pcf(range(1,85), 30, 30, xiform = 'log', corrections=['true'], cmap='hot')
    #plot_bao_cute2pcf(range(1,85), 30, 30, xiform = 'log', corrections=['true'], cmap='jet')
    #plot_bao_cute2pcf(range(1,85), 30, 30, xiform = 'asinh', corrections=['true'])
    #plot_bao_cute2pcf(range(1,85), 30, 30, xiform = 'asinh', corrections=['true'], cmap='jet')
