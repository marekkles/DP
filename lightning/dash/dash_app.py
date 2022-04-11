from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from jupyter_dash import JupyterDash
from typing import List, Union, Dict, Any, Callable, Optional, Tuple

def app_callbacks(app) -> None:
    """
    Creates callbacks for the given app.
    """
    
    return None

def app_factory(dash_class: Union[Dash, JupyterDash], server):
    """
    Creates a Dash app with the given server.
    """
    app = dash_class(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        routes_pathname_prefix='/dash/'
    )
    app.title = 'Lightning Dash'
    return app

