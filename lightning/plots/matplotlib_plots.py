import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy as sp
from scipy import stats

__all__ = ['matplotlib_plot_det', 'matplotlib_plot_dets', 
    'matplotlib_quality_performance', 'matplotlib_recognition_performance']

CM_TO_INCH = 1.0/2.5

def matplotlib_plot_det(fmr, fnmr, ax, label=""):
    x = np.array(stats.norm.ppf(fmr))
    y = np.array(stats.norm.ppf(fnmr))
    ax.plot(x, y, label=label)
    ticks = [0.004,0.01, 0.023, 0.05, 0.1, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = sp.stats.norm.ppf(ticks)
    tick_labels = [
        "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
        for s in ticks
    ]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-3, -2)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-3, -2)

def matplotlib_plot_dets(dets: dict, ax):
    for det in dets:
        matplotlib_plot_det(dets[det][0], dets[det][1], ax, det, f"Rejection Rate {det:.0%}")
    
def matplotlib_recognition_performance(fmr, fnmr, treashold, y_true, y_score):
    fig, [ax_det, ax_dist, ax_tr] = plt.subplots(1, 3, figsize=(12*CM_TO_INCH, 6*CM_TO_INCH))

    matplotlib_plot_det(fmr, fnmr, ax_det)
    ax_det.grid(alpha=0.8, linestyle='dashdot', lw=1)
    ax_det.set_title("FNMR@FMR")
    ax_det.set_xlabel("FMR")
    ax_det.set_ylabel("FNMR")
    ax_det.set_xscale('log')
    ax_det.set_yscale('log')
    ax_det.grid(alpha=0.8, linestyle='dashdot', lw=1)

    pair = y_score[y_true == 1]
    impostor = y_score[y_true == 0]
    impostor_hist = np.histogram(impostor,25,density=True)
    pair_hist = np.histogram(pair,25,density=True)

    ax_dist.hist(pair_hist[1][:-1], pair_hist[1], weights=pair_hist[0],alpha=0.73,label='Genuine')
    ax_dist.hist(impostor_hist[1][:-1], impostor_hist[1], weights=impostor_hist[0],alpha=0.73,label='Impostor')
    ax_dist.legend()
    ax_dist.set_xlabel("Score")
    ax_dist.set_ylabel("Probability")
    ax_dist.set_title("Impostor/Genuine distributions")

    ax_tr.plot(treashold, fmr, label="FMR")
    ax_tr.plot(treashold, fnmr, label="FNMR")
    ax_tr.grid(alpha=0.8, linestyle='dashdot', lw=1)
    ax_tr.legend()
    ax_tr.set_title("FNMR,FMR@Treashold")
    ax_tr.set_xlabel("Treashold")
    
    return fig

def matplotlib_quality_performance(irr, fmr, dets):
    fig, [ax_irr, ax_det] = plt.subplots(1, 2, figsize=(12*CM_TO_INCH, 6*CM_TO_INCH))

    ax_irr.plot(irr, fmr, label=f"False Match Rate {0.01:.0%}")
    ax_irr.grid()
    ax_irr.set_ylabel("False Non-match Rate")
    ax_irr.set_xlabel("Rejection Rate")
    ax_irr.xaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=0))
    ax_irr.yaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=1))
    ax_irr.set_title("FNMR@IRR")
    ax_irr.legend()

    matplotlib_plot_dets(dets, ax_det)
    ax_det.autoscale(False)
    ax_det.set_xlabel("False Match Rate")
    ax_det.set_ylabel("False Non-match Rate")
    ax_det.legend(loc="upper right")
    ax_det.set_title("FNMR@FMR")
    ax_det.grid()

    return fig