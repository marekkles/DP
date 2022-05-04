from lightning.plots import *
from lightning.dataset import *
from lightning.evaluation import *
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly as py
import numpy as np
import pandas as pd
import pickle
import yaml
import os
import re
import collections
import jupyter_dash


layout = html.Div([
    html.Div([
        html.H1("Masters thesis results"),
    ], style={'text-align': 'center', 'flex': 1}),
    html.Div([
        html.Div([
            html.H2("Options"),
            html.Label("Runs root directory",
                style={'width': '100%'}),
            html.Br(),
            dcc.Input(
                id='runs-input',
                placeholder='Enter path...',
                type='text',
                value='',
                style={'width': '100%'}
            ),
            html.Br(),
            html.Label("Runs",
                style={'width': '100%'}),
            dcc.Dropdown(
                id='runs-dropdown', clearable=True,
                value=None, options=[],
                style={'width': '100%'}),
            html.Label("Datasets",
                style={'width': '100%'}),
            dcc.Dropdown(
                id='datasets-dropdown', clearable=True,
                value=None, options=[],
                style={'width': '100%'}),
            html.Label("Metric",
                style={'width': '100%'}),
            dcc.Dropdown(
                id='metric-dropdown', clearable=True,
                value=None, options=[],
                style={'width': '100%'}),
            html.Label("Quality metric",
                style={'width': '100%'}),
            dcc.Dropdown(
                id='quality-metric-dropdown', clearable=True,
                value=None, options=[],
                style={'width': '100%'}),
            html.Label("Reject rate",
                style={'width': '100%'}),
            dcc.Slider(0, 30, id='quality-metric-slider', value=10),
            dcc.Input(id='save-quality-title', type='text', placeholder='Enter title of the quality plot'),
            html.Button('Save quality', id='save-quality-button'),
            html.Br(),
            dcc.Input(id='save-recognition-title', type='text', placeholder='Enter title of the recognition plot'),
            html.Button('Save recognition', id='save-recognition-button'),
            html.Br(),
            html.Label("", id='save-quality-status'),
            html.Label("", id='save-recognition-status'),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            html.H2(id='args-label', style={'width': '100%'}),
            html.Div([
                html.Samp(id='args-div')
            ], style={'height': '80%', 'overflow-y':'scroll'})
        ], style={'white-space': 'pre-wrap','height': '500px','max-height': '500px','padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'flex': 1,}),
    html.Div([
        dcc.Graph(
            id='quality-graph',
            figure=py.subplots.make_subplots(rows=1, cols=2)
        ),
        dcc.Graph(
            id='performance-graph',
            figure=py.subplots.make_subplots(rows=1, cols=3)
        ),dcc.Graph(
            id='training-graph',
            figure=py.subplots.make_subplots(rows=1, cols=2)
        ),
    ], style={'flex': 1, 'padding' : '10px'}),
    html.Div([
        html.H2('Combine results', style={'width': '100%'}),
        html.Label('Plots root directory', style={'width': '100%'}),
        dcc.Input(type='text', id='combine-path-input', placeholder='Enter path...', style={'width': '100%'}),
    ], style={'flex': 1, 'padding' : '10px'}),
], style={'display': 'flex', 'flex-direction': 'column'})

app_state = {
    "runs_root": None,
    "run_select":None,
    "dataset_select":None,
    "dataset":None,
    "args":None,
    "embedding":None,
    "scores":None,
    "quality":None,
    "fnmr_at_irr":None,
    "irr":None,
    "fnmr_at_fmr":None,
    "fmr":None,
    "quality_dets":None,
}

load_figure_template("bootstrap")
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
#app = jupyter_dash.JupyterDash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Load Data
app.layout = layout

@app.callback(
    Output(component_id='runs-dropdown', component_property='options'),
    Output(component_id='runs-dropdown', component_property='value'),
    Input(component_id='runs-input', component_property='value')
)
def load_runs_list(runs):
    global app_state

    app_state['runs_root'] = None
    app_state['args'] = None
    app_state["embedding"] = None
    app_state["dataset"] = None
    app_state["scores"] = None
    app_state["quality"] = None
    app_state["fnmr_at_irr"] = None
    app_state["irr"] = None
    app_state["fnmr_at_fmr"] = None
    app_state["fmr"] = None
    app_state["quality_dets"] = None

    if os.path.isdir(runs):
        dirs = os.listdir(runs)
        lt = [(d, os.path.join(runs, d)) for d in dirs if os.path.isfile(os.path.join(runs, d, 'args.yaml'))]
        app_state['runs_root'] = runs
    else:
        lt = []
    
    proj_dict={}
    for name, path in lt:
        with open(os.path.join(path, 'args.yaml')) as f:
            arg = yaml.load(f, yaml.FullLoader)
        if not arg['project_name'] in proj_dict:
            proj_dict[arg['project_name']] = []
        proj_dict[arg['project_name']].append((name, path))
    
    lt = []
    for project_name in proj_dict:
        lt.append({
            "label" : f"~~~ {project_name} ~~~",
            "value" : ''
        })
        for name, path in proj_dict[project_name]:
            lt.append({
                "label" : name,
                "value" : path
            })
    return lt, None

@app.callback(
    Output('datasets-dropdown', 'options'),
    Output('datasets-dropdown','value'),
    Output('args-div','children'),
    Output('args-label','children'),
    Output('training-graph', 'figure'),
    Input('runs-dropdown', 'value')
)
def load_run(input_value):
    global app_state

    app_state['args'] = None
    app_state["embedding"] = None
    app_state["dataset_select"] = None
    app_state["dataset"] = None
    app_state["scores"] = None
    app_state["quality"] = None
    app_state["fnmr_at_irr"] = None
    app_state["irr"] = None
    app_state["fnmr_at_fmr"] = None
    app_state["fmr"] = None
    app_state["quality_dets"] = None

    if input_value != None and input_value != '':
        lt = os.listdir(input_value)
        lt = list(filter(lambda x: x.startswith('embedding-'), lt))
        lt = [{
            'label': x.split('-')[-1].split('.')[0],
            'value': os.path.join(input_value, x)
        } for x in lt]
        with open(os.path.join(input_value, 'args.yaml')) as f:
            app_state["args"] = yaml.load(f, yaml.FullLoader)
            a = yaml.dump(app_state["args"], indent=2)
            l = app_state["args"]['project_name']
        df = pd.read_csv(os.path.join(input_value, 'csvs', 'metrics.csv'))
        df_train = df.dropna(subset=['train_acc', 'train_loss', 'step'])
        df_val = df.dropna(subset=['val_acc', 'val_loss', 'step', 'epoch'])
        fig = py.subplots.make_subplots(rows=1, cols=2)
        fig.add_traces(
            [
            go.Scatter(x=df_train['step'], y=df_train['train_acc'], name='train_acc'),
            go.Scatter(x=df_val['step'], y=df_val['val_acc'], name='val_acc'),
            go.Scatter(x=df_train['step'], y=df_train['train_loss'], name='train_loss'),
            go.Scatter(x=df_val['step'], y=df_val['val_loss'], name='val_loss'),
            ],
            rows=[1, 1,1,1],
            cols=[1, 1,2,2]
        )
    else:
        lt = []
        a = ''
        l = ''
        fig = py.subplots.make_subplots(rows=1, cols=2)
    return lt, None, a, l, fig

@app.callback(
    Output('metric-dropdown','options'),
    Output('metric-dropdown','value'),
    Input('datasets-dropdown','value')
)
def load_embedding(input_value):
    global app_state

    app_state["embedding"] = None
    app_state["dataset_select"] = None
    app_state["dataset"] = None
    app_state["scores"] = None
    app_state["quality"] = None
    app_state["fnmr_at_irr"] = None
    app_state["irr"] = None
    app_state["fnmr_at_fmr"] = None
    app_state["fmr"] = None
    app_state["quality_dets"] = None

    if input_value != None:
        ds = input_value.split('-')[-1].split('.')[0]
        app_state["dataset_select"] = ds

        if app_state["args"] != None and input_value != None and ds != 'iris_verification_pseudo':
            with open(input_value, 'rb') as f:
                app_state["embedding"] = pickle.load(f)
            app_state["dataset"] = verification_dataset_factory(
                os.path.join('../Datasets', ds),
                app_state["args"]["num_in_channels"],
                subset=None,
                transform=predict_transform(**app_state["args"]["predict_transform"]),
                autocrop=app_state["args"]["auto_crop"],
                unwrap=app_state["args"]["unwrap"],
            )
            return ["cosine", "euclidean", "cityblock"], None
        else:
            return [], None
    else:
        return [], None

@app.callback(
    Output(component_id='performance-graph', component_property='figure'),
    Output('quality-metric-dropdown', 'options'),
    Output('quality-metric-dropdown', 'value'),
    Input(component_id='metric-dropdown', component_property='value'),
)
def load_metric(input_value):
    global app_state

    app_state["scores"] = None
    app_state["quality"] = None
    app_state["fnmr_at_irr"] = None
    app_state["irr"] = None
    app_state["fnmr_at_fmr"] = None
    app_state["fmr"] = None
    app_state["quality_dets"] = None

    if input_value != None and  app_state["dataset"] != None:# and app_state["dataset_select"] != None:
        app_state["scores"] = pairs_impostor_scores(app_state["dataset"].pairs, app_state["dataset"].impostors, app_state["embedding"], input_value)
        labels, scores, pairs = generate_labels_scores(app_state["scores"]['pairs'], app_state["scores"]['impostors'])
        scores = -scores
        fmr, fnmr, treashold = det_curve(labels, scores)
        
        app_state["fnmr_at_fmr"] = fnmr
        app_state["fmr"] = fmr

        quality_ls = []
        #print (app_state["dataset_select"])
        for run in os.listdir(app_state["runs_root"]):
            for f in os.listdir(os.path.join(app_state["runs_root"], run)):
                qds = f.split('.')[0]
                if (qds.startswith('quality-') or qds.startswith('deviation-')) and qds.endswith(app_state["dataset_select"]):
                    quality_ls.append({
                        'label': run,
                        'value': os.path.join(app_state["runs_root"], run, f) 
                    })
        
        quality_ls.append({
            'label': 'magnitude',
            'value': 'magnitude'
        })
        #print(quality_ls)

        return plotly_recognition_performance(fmr, fnmr, treashold, labels, scores), quality_ls, None
    else:
        return py.subplots.make_subplots(rows=1, cols=3), [], None
        

@app.callback(
    Output(component_id='quality-metric-slider', component_property='value'),
    Input(component_id='quality-metric-dropdown', component_property='value')
)
def load_quality(input_value):
    global app_state

    app_state["quality"] = None
    app_state["fnmr_at_irr"] = None
    app_state["irr"] = None
    app_state["quality_dets"] = None

    if input_value != None and app_state["embedding"] != None and app_state["scores"] != None:
        if input_value == 'magnitude':
            #print("magnitude")
            app_state["quality"] = {}
            for p in app_state["embedding"]:
                app_state["quality"][p] = np.linalg.norm(app_state["embedding"][p])
        else:
            #print(f"Auxilary: {input_value}")
            with open(input_value, 'rb') as f:
                app_state["quality"] = pickle.load(f)
            if input_value.split('/')[-1].split('-')[0] == 'deviation':
                app_state["quality"] = {k: np.average(v) for k, v in app_state["quality"].items()}
            for q in app_state["quality"]:
                if type(app_state["quality"][q]) == np.ndarray:
                    app_state["quality"][q] = app_state["quality"][q].item()
            if input_value.split('/')[-1].split('-')[1] == 'DfsNet':
                app_state["quality"] = {k: v for k, v in app_state["quality"].items()}
        
            
        labels, scores, quality_scores = generate_sorted_labels_scores_quality(app_state["scores"]['pairs'], app_state["scores"]['impostors'], app_state["quality"])
        scores = -scores
        irr, fnmr = eer_at_irr(labels, scores, max_reject_rate=0.2)
        app_state["fnmr_at_irr"] = fnmr
        app_state["irr"] = irr
        return 10
    else:
        return 0

@app.callback(
    Output(component_id='quality-graph', component_property='figure'),
    Input(component_id='quality-metric-slider', component_property='value')
)
def set_quality(input_value):
    global app_state

    app_state["quality_dets"] = None

    if app_state["quality"] != None and app_state["fnmr_at_irr"] != None and app_state["irr"] != None and app_state["scores"] != None:
        labels, scores, quality_scores = generate_sorted_labels_scores_quality(app_state["scores"]['pairs'], app_state["scores"]['impostors'], app_state["quality"])
        scores = -scores
        app_state["quality_dets"] = dict(det_for_irrs(labels, scores, [0, input_value/100.0]))
        plt = plotly_quality_performance(app_state["irr"], app_state["fnmr_at_irr"], app_state["quality_dets"])
    else:
        plt = py.subplots.make_subplots(rows=1, cols=2)
    return plt

@app.callback(
    Output(component_id='save-recognition-status', component_property='children'),
    [Input(component_id='save-recognition-button', component_property='n_clicks')],
    [State(component_id='save-recognition-title', component_property='value')],
)
def save_recognition(n_clicks, title):
    global app_state

    if n_clicks != None and n_clicks > 0 and not (app_state["fnmr_at_fmr"] is None) and not (app_state["fmr"] is None) and title != None:
        with open(os.path.join(
            'csvs', 'recognition', 
            title.strip() + '.pickle'
        ), 'wb') as f:
            pickle.dump({
                'title' : title,
                'dataset': app_state["dataset_select"],
                'fnmr_at_fmr': app_state["fnmr_at_fmr"],
                'fmr': app_state["fmr"]
            }, f)
        return 'Saved ' + title + ' ' + app_state['dataset_select']
    else:
        return ''
    
@app.callback(
    Output(component_id='save-quality-status', component_property='children'),
    [Input(component_id='save-quality-button', component_property='n_clicks')],
    [State(component_id='save-quality-title', component_property='value')],
)
def save_quality(n_clicks, title):
    global app_state

    if n_clicks != None and n_clicks > 0 and not (app_state["fnmr_at_irr"] is None) and not (app_state["irr"] is None) and not  (app_state['quality_dets'] is None) and title != None:
        with open(os.path.join(
            'csvs', 'quality', 
            title.strip() + '.pickle'
        ), 'wb') as f:
            pickle.dump({
                'title' : title,
                'dataset': app_state["dataset_select"],
                'fnmr_at_irr': app_state["fnmr_at_irr"],
                'irr': app_state["irr"],
                'dets': app_state['quality_dets']
            }, f)
        return 'Saved ' + title + ' ' + app_state['dataset_select']
    else:
        return ''

@app.callback(
    Output(component_id='recognition-plot-dropdown', component_property='options'),
    Output(component_id='recognition-plot-dropdown', component_property='value'),
    Output(component_id='quality-plot-dropdown', component_property='options'),
    Output(component_id='quality-plot-dropdown', component_property='value'),
    Input(component_id='combine-path-input', component_property='value')
)
def load_plots(input_value):
    if not (input_value is None) and os.path.isdir(input_value) and os.path.isdir(os.path.join(input_value, 'quality')) and os.path.isdir(os.path.join(input_value, 'recognition')):
        recogntion_list = [
            {
                'label': f.split('.')[0],
                'value': os.path.join(input_value, 'recognition', f)
            }
            for f in os.listdir(os.path.join(input_value, 'recognition'))
        ]
        quality_list = [
            {
                'label': f.split('.')[0],
                'value': os.path.join(input_value, 'quality', f)
            }
            for f in os.listdir(os.path.join(input_value, 'quality'))
        ]
        return recogntion_list, None, quality_list, None
    else:
        return [], None, [], None


#app.run_server(mode='inline', height=2100, port=8083)
app.run_server(port=8080)