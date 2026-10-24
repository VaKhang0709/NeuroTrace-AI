"""V6 fixed-rate signal pipeline. This is a NEW contract, not V5-compatible.

Inputs: schema Trajectory, explicit seconds. No stroke concatenation here. Call
separately per stroke if acquisition records have pen-up boundaries. Samples
must span >=5.11s for one full 512-step window; no duration-stretching/padding.
"""
import numpy as np
import pywt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from src.data_loaders.schema import DOMAINS, TASKS

FEATURES = ('x','y','pressure','vx','vy','speed','ax','ay','acceleration','jerk',
            'curvature','pressure_derivative','azimuth_derivative',
            'tremor_x_4_8','tremor_y_4_8','phys_x_8_12','phys_y_8_12',
            'pressure_observed','azimuth_observed')
CUTOFF = dict(wacom=15., touch=10., mouse=12., stylus=15., unknown=10.)
BINARY_CHANNELS = (17, 18)


def virtual_pressure(speed, radius):
    """V5-inspired full-stroke proxy, NOT a measured force or clinical label."""
    v = (speed-speed.min())/(np.ptp(speed)+1e-6)
    return .65*(1-.35*v)+.15*np.clip(radius,0,1.5)


def wavelet_band(a, lo, hi, fs=100.):
    """db4 level-5 DWT reconstruction + FFT boundary selection for exact Hz bands.

    Pure dyadic DWT bands do NOT equal 4-8/8-12 Hz at 100Hz. Keep overlapping
    details then select frequencies. Finite-stroke boundary artifacts remain.
    """
    coefficients=pywt.wavedec(a,'db4',level=5,mode='symmetric')
    selected=[np.zeros_like(c) for c in coefficients]
    for i in range(1,len(coefficients)):
        level=6-i
        low,high=fs/(2**(level+1)),fs/(2**level)
        if high>lo and low<hi:
            selected[i]=coefficients[i]
    reconstructed=pywt.waverec(selected,'db4',mode='symmetric')[:len(a)]
    spectrum=np.fft.rfft(reconstructed)
    f=np.fft.rfftfreq(len(a),1/fs)
    spectrum[(f<lo)|(f>=hi)]=0
    return np.fft.irfft(spectrum,n=len(a))


def features(trial):
    trial.validate()
    a=trial.samples; t=a[:,2]-a[0,2]
    rate=1/np.median(np.diff(t))
    if rate<25:
        raise ValueError('Sampling rate <25Hz cannot support requested physiology bands')
    grid=np.arange(int(np.floor(t[-1]*100))+1,dtype=float)/100
    if len(grid)<512:
        raise ValueError('Need >=512 canonical samples; short trial excluded, not time-stretched')
    cutoff=min(CUTOFF[trial.domain],.45*rate)
    sos=butter(4,cutoff,fs=100,output='sos')
    def channel(v):
        return sosfiltfilt(sos,CubicSpline(t,v)(grid))
    x,y=channel(a[:,0]),channel(a[:,1])
    x-=np.median(x); y-=np.median(y)
    spatial=np.percentile(np.hypot(x,y),95)
    if spatial<1e-8:
        raise ValueError('Stationary trajectory')
    x/=spatial; y/=spatial
    vx,vy=np.gradient(x,.01),np.gradient(y,.01)
    speed=np.hypot(vx,vy)
    finite=np.isfinite(a[:,3]); has_pressure=bool(finite.sum()>=10 and np.unique(a[finite,3].round(4)).size>=4)
    if has_pressure:
        pv=np.interp(t,t[finite],a[finite,3]); p=channel(pv)
    else:
        p=virtual_pressure(speed,np.hypot(x,y))
    finite_a=np.isfinite(a[:,4]); has_azimuth=bool(finite_a.sum()>=10)
    az=channel(np.interp(t,t[finite_a],np.unwrap(a[finite_a,4]))) if has_azimuth else np.zeros_like(x)
    ax,ay=np.gradient(vx,.01),np.gradient(vy,.01)
    acc=np.hypot(ax,ay); jerk=np.hypot(np.gradient(ax,.01),np.gradient(ay,.01))
    curvature=(vx*ay-vy*ax)/np.maximum((vx*vx+vy*vy)**1.5,1e-8)
    channels=[x,y,p,vx,vy,speed,ax,ay,acc,jerk,np.clip(curvature,-1e3,1e3),
              np.gradient(p,.01),np.gradient(az,.01),wavelet_band(x,4,8),
              wavelet_band(y,4,8),wavelet_band(x,8,12),wavelet_band(y,8,12),
              np.full_like(x,has_pressure),np.full_like(x,has_azimuth)]
    result=np.stack(channels)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite derived features')
    return result,dict(original_rate_hz=float(rate),cutoff_hz=cutoff,canonical_rate_hz=100)


def normalize_windows(x):
    """Channel-wise per-window population z-score; binary flags retained.

    Unlike V5's geometry/pressure/dt transforms this is V6-only. Constant signal
    channels become zero, binary availability is not z-scored away.
    """
    out=np.array(x,dtype=np.float64,copy=True)
    continuous=[i for i in range(len(FEATURES)) if i not in BINARY_CHANNELS]
    values=out[:,continuous,:]
    out[:,continuous,:]=(values-values.mean(-1,keepdims=True))/(values.std(-1,keepdims=True)+1e-6)
    return out.astype(np.float32)


def window_features(values, stride=128):
    if stride<1 or len(values.shape)!=2 or values.shape[0]!=len(FEATURES):
        raise ValueError('Bad features/stride')
    n=values.shape[-1]
    if n<512:
        raise ValueError('Short trajectory')
    starts=list(range(0,n-511,stride))
    if starts[-1]!=n-512: starts.append(n-512)
    return np.stack([values[:,s:s+512] for s in starts])


def transform(trial,stride=128):
    values,qc=features(trial)
    return normalize_windows(window_features(values,stride)),qc
