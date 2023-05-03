# TO DO:
#
#THIS CODE DOESN'T USE THE ALTITUDE INFORMATION IN THE PLCal30-BMCODE_plline-###.txt FILES THAT IT MAKES...
#
# 1) IMPLEMENT TIKHONOV REGULARIZATION TO THE FIT
#    https://en.wikipedia.org/wiki/Tikhonov_regularization
# 2) Rewrite this code to use ini files and/or run more automatically than it currently does.
# 3) Add HDF5 file output!!!
#
#

import datetime
import os
import numpy
import numpy as np
import scipy
from matplotlib import pyplot
import tables
from scipy.optimize import leastsq
import scipy.io as sio
import glob

# CHANGE FROM HERE:


# The experiment directory /Volumes/AMISR_PROCESSED/calibration/AMISR/calibration_PFISR/yyyymm/
# Use yyyymmdd.000 if ksys is from multiple experiments
#exp_dir = '20210908.004'
#exp_dir = '20220215.001'
#exp_dir = '20221222.012'
exp_dir = '20230301.004'
radarmode = 'PLCal30_7down7up'
exp_dir = '20230316.003'
exp_dir = '20230330.003'
exp_dir = '20230414.003'
exp_dir = '20230429.002'
radarmode = 'PLCal30_5down7up'
syear=exp_dir[:4]
smonth=exp_dir[4:6]

yyyymm = exp_dir[:6]
exp = 'cal-%s' %(yyyymm)

# scalar to make fitting more well conditioned
scalar = 1.e19

# Get a list of the fitted data files for the plasma line calibration mode
#alternating code files
#filelist_ac = ['/Volumes/AMISR_PROCESSED/processed_data/PFISR/2021/09/PLCal30/20210923.002/20210923.002_ac_5min.h5'
#              ]
filelist_ac = [f'/Volumes/AMISR_PROCESSED/processed_data/PFISR/{syear}/{smonth}/{radarmode}/{exp_dir}/{exp_dir}_ac_5min.h5'
        ]

print("filelist_ac",filelist_ac)
#alternating code files
#filelist_lp = ['/Volumes/AMISR_PROCESSED/processed_data/PFISR/2021/09/PLCal30/20210923.002/20210923.002_lp_5min.h5'
#              ]
filelist_lp = [f'/Volumes/AMISR_PROCESSED/processed_data/PFISR/{syear}/{smonth}/{radarmode}/{exp_dir}/{exp_dir}_lp_5min.h5'
              ]
print("filelist_lp",filelist_lp)

# which pulse type to use for the Ksys files? Options are 'ac','lp', or 'both'
pulse_to_use = 'both'

# Output directory for Ksys files and plots
#output_calpath = '/Volumes/AMISR_PROCESSED/calibration/AMISR/calibration_PFISR/'+yyyymm+'/PLCal30/'+exp_dir
output_calpath = os.path.join('/Volumes/AMISR_PROCESSED/calibration/AMISR/calibration_PFISR',yyyymm,radarmode,exp_dir)
#output_calpath = '/home/asreimer/calibration/ksys_output_dir/'+yyyymm+'/PLCal30/'+exp_dir


## DO NOT CHANGE BELOW HERE

# Used for plots and ksys file names
now = datetime.datetime.now()
date = now.strftime("%m.%d.%Y")


# Az El converted to angle from boresite
def get_BS_angle(in_az,in_el):
    az = in_az*scipy.pi/180.0
    el = in_el*scipy.pi/180.0
    az_bs = 15.0*scipy.pi/180.0
    el_bs = 74.0*scipy.pi/180.0
    k = numpy.array([[scipy.cos(el)*scipy.cos(az)],
                     [scipy.cos(el)*scipy.sin(az)],
                     [scipy.sin(el)]])
    
    tk = rotmat(k,3,az_bs)
    tk2 = rotmat(tk,2,scipy.pi/2.0-el_bs)
    
    alphaBS=90.0-scipy.arcsin(tk2[2])*180.0/scipy.pi
    
    return alphaBS


# Use these files to determine which beamcodes we have available from all the files. 
# We will also use the location of these fitted files to determine the list of plline*.txt files
def beamcodes_and_plline_files(pulsetype,fitted_filelist,exclude=[]):
    beamcodes_calfiles = list()
    beamcodes = list()
    for f in fitted_filelist:
        filename = os.path.basename(f)
        filepath = os.path.dirname(f)
        expdir = os.path.basename(filepath)

        # Get plline filelist
        search_path = os.path.join(filepath,'cal-0/*plline*%s*.txt' % (pulsetype))
        cal_txt_files = glob.glob(search_path)
        if len(cal_txt_files) == 0:
            continue

        beamcodes_calfiles.extend(cal_txt_files)

        # Get beamcode information
        with tables.open_file(f,'r') as h5:
            BM = np.array(h5.root.BeamCodes.read())

        num_beams,_ = BM.shape
        for i in range(num_beams):
            bmcode = int(BM[i,0])
            az = float(BM[i,1])
            el = float(BM[i,2])
            ksys = float(BM[i,3])
            if not bmcode in exclude:
                beamcodes.append((bmcode,az,el,ksys))

    return beamcodes_calfiles,np.array(sorted(list(set(beamcodes))))


def get_corrected_ksys(beamcodes_calfiles,beamcodes):
    cal_beams = dict()
    for fname in beamcodes_calfiles:
        with open(fname,'r') as f:
            bmcode = int(float(f.readline().split()[0]))
            temp = f.readline().split()
            ksys_correction_ratio = float(temp[0])
            ksys_correction_ratio_error = float(temp[1])

        if not bmcode in beamcodes:
            continue

        if not bmcode in cal_beams.keys():
            cal_beams[bmcode] = list()

        bm_ind = np.where(bmcode==beamcodes[:,0])[0]
        az      = float(beamcodes[bm_ind,1])
        el      = float(beamcodes[bm_ind,2])
        angle_from_boresite = float(get_BS_angle(az,el))

        oldksys              = float(beamcodes[bm_ind,3])
        corrected_ksys       = oldksys*ksys_correction_ratio
        corrected_ksys_error = corrected_ksys*ksys_correction_ratio_error

        cal_beams[bmcode].append([angle_from_boresite,corrected_ksys,corrected_ksys_error])

    # Now that we've read every file and calculated the corrected ksys values from each
    # we need to combine any that are part of the same beam
    for bm in cal_beams.keys():
        if len(cal_beams[bm]) > 1:
            bs_angle         = float(cal_beams[bm][0][0])
            total_ksys       = np.array([x[1] for x in cal_beams[bm]])
            total_ksys_error = np.array([x[2] for x in cal_beams[bm]])

            mean_ksys = np.nanmean(total_ksys)
            mean_ksys_error = np.sqrt(np.nanmean(total_ksys_error**2))  # RMS

            cal_beams[bm] = [bs_angle,mean_ksys,mean_ksys_error]
        else:
            cal_beams[bm] = cal_beams[bm][0]

    # filter out beams that have nans in them for mean_ksys and mean_ksys_error
    remove_beams = list()
    for bm in cal_beams.keys():
        if np.any(np.isnan(cal_beams[bm])):
            remove_beams.append(bm)
    for bm in remove_beams:
        del cal_beams[bm]


    return cal_beams


def fit_cosine_gain_model(cal_beams,use='abc',fixed=None):
    x  = list()
    y  = list()
    ey = list()
    for bm in cal_beams.keys():
        x.append(cal_beams[bm][0])
        y.append(cal_beams[bm][1])
        ey.append(cal_beams[bm][2])

    x  = np.array(x)
    x.astype('float64')
    y  = np.array(y)*scalar
    y.astype('float64')
    ey = np.array(ey)*scalar
    ey.astype('float64')

    if use == 'abc':
        [a,b,c],flag = leastsq(residual,[1.8,4.1,0.0],args=(y,ey,x))
    if use == 'ac':
        [a,c],flag = leastsq(residual_ac,[1.8,-0.1],args=(y,ey,x,fixed))
        b=fixed
    if use == 'ab':
        [a,b],flag = leastsq(residual_ab,[1.8,-0.1],args=(y,ey,x,fixed))
        c=fixed
    if use == 'bc':
        [b,c],flag = leastsq(residual_bc,[5.,-0.1],args=(y,ey,x,fixed))
        a=fixed
    if use == 'a':
        a,flag = leastsq(residual_a,1.8,args=(y,ey,x,fixed[0],fixed[1]))
        b,c = fixed
    return a,b,c


def write_new_ksys_txt(ac_abc,lp_abc,exp,date,use='ac'):
    base_beamcode_map = np.loadtxt('BeamCodeMap.txt')
    num_beams = np.shape(base_beamcode_map)[0]
    
    name1 = os.path.join(output_calpath,'%s-calibration-scalar-%s.txt' %(exp,date))
    name2 = os.path.join(output_calpath,'%s-calibration-ksys-%s.txt' %(exp,date))
    with open(name1,'w') as fid:
        with open(name2,'w') as fid2:
    
            for ibm in range(num_beams):
                tbm = base_beamcode_map[ibm,:]
                az = base_beamcode_map[ibm,1]
                el = base_beamcode_map[ibm,2]
                
                kold = base_beamcode_map[ibm][3]
                
                alphaBS = get_BS_angle(az,el)

                if use == 'ac' or use == 'both':
                    ksys_ac = ac_abc[0]*np.power(np.cos(alphaBS*np.pi/180.0+ac_abc[2]),ac_abc[1])/scalar
                    ksys = ksys_ac

                if use == 'lp' or use == 'both':
                    ksys_lp = lp_abc[0]*np.power(np.cos(alphaBS*np.pi/180.0+lp_abc[2]),lp_abc[1])/scalar
                    ksys = ksys_lp

                if use == 'both':
                    ksys = (ksys_lp+ksys_ac) / 2.

                ksysCorr = ksys/kold
                
                #print tbm[0],tbm[1],tbm[2],ksys,ksysCorr
                fid.write('%d %2.2f %2.2f %2.2e %3.5f\n' %(tbm[0],tbm[1],tbm[2],ksys,ksysCorr))
                fid2.write('%d %2.2f %2.2f %2.2e\n'%(tbm[0],tbm[1],tbm[2],ksys))


def make_ksys_plot(abc,cal_beams,pulsetype):

    if pulsetype=='ac':
        title = r'Alternating Code Derived K$_{sys}$: a=%2.3f, b=%2.3f, c=%2.3f' % (abc[0],abc[1],abc[2])
    if pulsetype=='lp':
        title = r'Long Pulse Derived K$_{sys}$: a=%2.3f, b=%2.3f, c=%2.3f' % (abc[0],abc[1],abc[2])

    # plot the model fitted ksys
    alphaBS = np.linspace(0,40.0,num=100)
    model_ksys = abc[0]*np.power(np.cos(alphaBS*np.pi/180.0+abc[2]),abc[1])

    fig = pyplot.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(alphaBS,model_ksys,'k')
    ax.plot(alphaBS,model_ksys*0.9,'k--')
    ax.plot(alphaBS,model_ksys*1.1,'k--')

    # plot the ksys from the fitted files
    x  = list()
    y  = list()
    ey = list()
    bmnum = list()
    for bm in cal_beams.keys():
        bmnum.append(bm)
        x.append(cal_beams[bm][0])
        y.append(cal_beams[bm][1])
        ey.append(cal_beams[bm][2])

    bmnum = np.array(bmnum)
    x  = np.array(x)
    y  = np.array(y)*scalar
    ey = np.array(ey)*scalar

    ymax = np.max(y)

    num_beams = len(bmnum)
    for i in range(num_beams):
        ax.plot(x[i],y[i],'k.',marker='o')
        ax.plot([x[i],x[i]],[y[i]-ey[i],y[i]+ey[i]],'b')
        ax.text(x[i],1.1*y[i],str(int(bmnum[i])),ha='left',va='top',color='red',zorder=100)
    ax.set_title(title)
    ax.set_xlabel('Angle off Boresight')
    ax.set_ylabel(r'K$_{sys}$ $\times$ %.1e' % (1.0/scalar))
    oname = '%s-%s-%s' % (exp,pulsetype,date)
    savepath = os.path.join(output_calpath,oname)
    fig.savefig(savepath + '.png', dpi=200)


def rotmat(input, dir, angle):
    if dir == 1:
        rotmat = numpy.array([ [1,0,0],
                               [0, scipy.cos(angle), scipy.sin(angle)],
                               [0, -scipy.sin(angle), scipy.cos(angle)]])
    if dir == 2:
        rotmat = numpy.array([ [scipy.cos(angle), 0, -scipy.sin(angle)],
                               [0, 1, 0],
                               [scipy.sin(angle), 0, scipy.cos(angle)]])
    if dir == 3:
        rotmat = numpy.array([ [scipy.cos(angle), scipy.sin(angle), 0],
                               [-scipy.sin(angle), scipy.cos(angle), 0],
                               [0, 0, 1]])
    
    return scipy.dot(rotmat,input)
    
    
def func(x,a,b,c):
    temp = a*scipy.power(scipy.cos(x*scipy.pi/180.0+c), b)
    return temp


def residual(p, y, ey, x):
    a,b,c = p
    res = (y - func(x,a,b,c))/ey
    res = res.astype('float64')
    return res

def residual_bc(p, y, ey, x, a):
    b,c = p
    return (y - func(x,a,b,c))/ey

def residual_ab(p, y, ey, x, c):
    a,b = p
    return (y - func(x,a,b,c))/ey

    
def residual_ac(p, y, ey, x, b):
    a,c = p
    return (y - func(x,a,b,c))/ey

def residual_a(p, y, ey, x, b, c):
    a = p
    return (y - func(x,a,b,c))/ey

    
if __name__ == '__main__':

    # arbitrary base model (based on NEC simulations? Who knows... apparently documenting anything was novel to previous employees...)
    # angle_from_boresite = np.linspace(0,40,100)
    # A = 0.98395
    # B = 3.8781
    # oldksys = A * np.power(np.cos(angle_from_boresite*np.pi/180.0),B)

    # Get a list of all the plline files generated using run_plasmaline* script
    # Also get a list of all the beam codes that all of the files have
    beamcodes_ac_calfiles,beamcodes_ac = beamcodes_and_plline_files('ac',filelist_ac,exclude=[])
    beamcodes_lp_calfiles,beamcodes_lp = beamcodes_and_plline_files('lp',filelist_lp,exclude=[])

    # Parse the plline files and determine the corrected ksys values for each beam from the list of files
    ac_cal_beams = get_corrected_ksys(beamcodes_ac_calfiles,beamcodes_ac)
    lp_cal_beams = get_corrected_ksys(beamcodes_lp_calfiles,beamcodes_lp)

    # Now fit the cosine gain model to the corrected ksys to get a fitted ksys
    ac_abc = fit_cosine_gain_model(ac_cal_beams,'abc')
    lp_abc = fit_cosine_gain_model(lp_cal_beams,'abc')
    # ac_abc = fit_cosine_gain_model(ac_cal_beams,'a',fixed=[7.261,-0.088])
    # lp_abc = fit_cosine_gain_model(lp_cal_beams,'a',fixed=[7.261,-0.088])
    # ac_abc = fit_cosine_gain_model(ac_cal_beams,'ac',fixed=7.261)
    # lp_abc = fit_cosine_gain_model(lp_cal_beams,'ac',fixed=7.261)
    # ac_abc = fit_cosine_gain_model(ac_cal_beams,'ab',fixed=-0.167)
    # lp_abc = fit_cosine_gain_model(lp_cal_beams,'ab',fixed=-0.167)

    # Generate the new ksys files based on the fitted ksys curve
    write_new_ksys_txt(ac_abc,lp_abc,exp,date,use=pulse_to_use)

    # Make plots of the ksyses comparing the fitted, corrected, and uncorrected.
    make_ksys_plot(ac_abc,ac_cal_beams,'ac')
    make_ksys_plot(lp_abc,lp_cal_beams,'lp')

                
