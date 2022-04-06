import numpy as np
import plotly as py
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy as sp

__all__ = ['plotly_recognition_performance', 'plotly_quality_performance']

def plotly_det_scatter(fmr, fnmr,name="DET"):
    hovertext=[f"FMR: {x*100:.3f}% FNMR: {y*100:.3f}%" for (x, y) in zip(fmr, fnmr)]
    return go.Scatter(
        name=name, 
        x = sp.stats.norm.ppf(fmr),
        y = sp.stats.norm.ppf(fnmr),
        hovertext=hovertext,
        hoverinfo="text"
    )

def plotly_set_det_plot(fig, row=None, col=None):
    ticks = [0.001, 0.004,0.01, 0.023, 0.05, 0.1, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_vals = sp.stats.norm.ppf(ticks)
    tick_labels = [
        "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
        for s in ticks
    ]
    if row != None and col != None:
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_labels,
            row=row, col=col
        )
        fig.update_yaxes(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_labels,
            row=row, col=col
        )
    else:
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_labels,
        )
        fig.update_yaxes(
            tickmode = 'array',
            tickvals = tick_vals,
            ticktext = tick_labels
        )


def plotly_recognition_performance(fmr, fnmr, treashold, y_true, y_score):
    fig = py.subplots.make_subplots(rows=1, cols=3)
    hovertext=[f"FMR: {x:.3f} FNMR: {y:.3f}" for (x, y) in zip(fmr, fnmr)]
    fig.add_trace(plotly_det_scatter(fmr, fnmr, name="DET"), row=1, col=1)
    plotly_set_det_plot(fig, row=1, col=1)
    fig.add_trace(go.Histogram(name=f"Genuine", x=y_score[y_true == 1], histnorm='probability',opacity=0.5,nbinsx=20), row=1, col=2)
    fig.add_trace(go.Histogram(name=f"Impostor", x=y_score[y_true == 0], histnorm='probability',opacity=0.5,nbinsx=20), row=1, col=2)
    fig.update_layout(barmode='overlay')
    fig.add_trace(go.Scatter(name=f"FMR", y=fmr, x=treashold), row=1, col=3)
    fig.add_trace(go.Scatter(name=f"FNMR", y=fnmr, x=treashold), row=1, col=3)
    return fig

def plotly_quality_performance(irr, fnmr, dets):
    irr=np.array(irr)
    fnmr=np.array(fnmr)
    fig = py.subplots.make_subplots(rows=1, cols=2)
    for d in dets:
        fig.add_trace(plotly_det_scatter(dets[d][0], dets[d][1],
            name=f"Rejection rate {d:.0%}"
        ), row=1, col=2)
    plotly_set_det_plot(fig, row=1, col=2)
    fig.add_trace(go.Scatter(
        name=f"False match rate {0.01:.0%}",
        x=irr*100, y=fnmr*100, hovertemplate="IRR: %{x:0.3f}% FNMR: %{x:0.3f}%"),row=1, col=1)

    return fig