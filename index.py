###
# App Name:  Visual BIM
# App URI: https://opendatabim.io/
# Description: Creation of parametric visualization of RVT and IFC files through a CSV file
# Author: Artem Boiko
# Version:  1.1.4
# OpenDataBIM
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
###

#from app import app
import base64
import datetime
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import io
import numpy as np
import plotly.express as px
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
import plotly.graph_objects as go


# Selecting data by category and converting string values to numeric values
def f_cat(dd_cat, dfi):

    # Selecting a category to filter
    if dd_cat == 'All categories':
        df = dfi
    else:
        df = dfi[dfi['Category'] == dd_cat]

    # Finding all columns with volumetric values
    volcolumns = []
    volname = ['[vV]olume', '[aA]rea', '[W]eight', '[hH]eight', '[lL]ength']
    for col in df.columns:
        for voln in volname:
            if re.search(voln, col):
                volcolumns.append(col)
            else:
                0
    propstr = volcolumns
    dfc = df

    # Converting all string values to numeric values
    def find_number(text):
        num = re.findall(r'[0-9]+', text)
        return ".".join(num)
    for el in propstr:
        dfc[el] = dfc[el].astype(str)
        dfc[el] = dfc[el].apply(lambda x: find_number(x))
        dfc[el] = dfc[el].fillna(0)
        dfc[el] = pd.to_numeric(dfc[el], errors='coerce')
        dfc[el] = dfc[el].replace(np.nan, 0)
        dfc[el] = dfc[el].replace('None', 0)
        dfc[el] = dfc[el].fillna(0)
        try:
            dfc[el] = dfc[el].astype(float)
        except:
            pass
    df = df.drop('Category', axis=1)
    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass

    return dfc, df

# Splitting into numeric and categorical


def cr_df_floatstring(dd_volspec, dd_groupval, dd_opval, dc_forms1, dc_forms2, dfc, df):

    # Breakdown of the dataframe into those where there are only numeric values or where only text values are
    dfstring = dfc.select_dtypes(include=['object'])
    dffloat = dfc.select_dtypes(include=['float64'])

    # Cleaning and averaging
    dffloat = dffloat[dffloat.columns[dffloat.max() > 0]]
    dffloat[dd_groupval] = dfstring[dd_groupval]
    dfstring_gr = dfstring.groupby([dd_groupval]).first()
    dffloat_gr = dffloat.groupby([dd_groupval]).mean()
    df_vol = df.groupby([dd_groupval])[dd_volspec].agg([dd_opval, 'count'])

    # Removing columns indicating the type and category of elements
    names = ['Category', 'Type', 'Family and Type',
             'Type Id', 'Type Name', 'Family Name']
    for el in names:
        try:
            dfstring_gr = dfstring_gr.drop(el, axis=1)
        except:
            pass
    dfstring_gr = dfstring_gr.select_dtypes(include=[object])
    dfstring_gr = dfstring_gr.astype(str)

    # Scalar string data
    le = preprocessing.LabelEncoder()
    dfstring_gr_2 = dfstring_gr.apply(lambda col: le.fit_transform(
        col.astype(str)), axis=0, result_type='expand')
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(dfstring_gr_2)
    dfstring_gr_2.loc[:, :] = scaled_values
    try:
        dffloat_gr = dffloat_gr.drop([0])
        dfstring_gr_2 = dfstring_gr_2.drop([0])
        dfstring_gr = dfstring_gr.drop([0])
    except:
        pass

    # Formation of dataframes for coordinates
    dfgall = dfstring_gr_2
    volmax = df_vol[dd_opval].max()
    countmax = df_vol['count'].max()
    df_vol[dd_opval] = df_vol[dd_opval].round(4)
    grname = dfgall.index.values
    dfgall = dfgall.reset_index(drop=True)

    # Getting coordinates XYZ from Categorical dataframes
    x, y = np.meshgrid(range(0, len(dfgall.columns)), dfgall.index)
    z = dfgall.values
    sizep = df_vol[dd_opval].values
    count = df_vol['count'].values
    z2 = dfstring_gr.values
    xn = []
    yn = []
    zn = []
    zn2 = []
    sizepn = []
    countn = []
    xyz = []
    grnamen = []
    for i, el in enumerate(z):
        zn.append([])
        zn2.append([])
        yn.append([])
        xn.append([])
        for i2, val in enumerate(el):
            if 1 == 1:
                zn[i].append(val)
                zn2[i].append(z2[i][i2])
                yn[i].append(y[i][i2])
                xn[i].append(x[i][i2])
        sizepn.append(sizep[i])
        countn.append(count[i])
        grnamen.append(grname[i])
    znf = []
    ynf = []
    xnf = []
    zn2f = []
    sizepnf = []
    countnf = []
    grnamenf = []
    for el in zn:
        for ue in el:
            znf.append(ue)
    for el in zn2:
        for ue in el:
            zn2f.append(ue)
    for el in xn:
        for ue in el:
            xnf.append(ue)
    for i, el in enumerate(yn):
        for val in el:
            ynf.append(val)
            sizepnf.append(sizepn[i])
            countnf.append(countn[i])
            grnamenf.append(grnamen[i])
    countnf = [1 if x == 0 else x for x in countnf]
    my_xticks = dfgall.columns
    dfgarray = dfgall.to_numpy()
    dfgarray = dfgarray.flatten()
    arrayall = np.array([znf, zn2f, ynf, xnf, sizepnf, countnf, dfgarray])
    arraynames = np.array([grnamenf])
    arrayall = np.transpose(arrayall)
    arraynames = np.transpose(arraynames)

    # Formation of a summary table with coordinates for all categorical data
    dfall = pd.DataFrame(data=arrayall, columns=[
                         'z', 'z2', 'y', 'x', 'The total volume', 'Quantity of elements', 'Value'], index=arraynames)
    dfall['y2'] = dfall.index
    dfnames = pd.DataFrame(data=arraynames, columns=['names'])
    multsizepnf = []
    for el in sizepnf:
        if el == 0:
            multsizepnf.append(el*volmax*10)
        else:
            multsizepnf.append(el*volmax)
    countnf = [element * countmax for element in countnf]
    a = np.array(dfstring_gr_2.columns)
    dfall['x2'] = np.resize(a, dfall.shape[0])
    dfall['Typ'] = dc_forms1

    # Create DataFrame for Float values and scale data
    dfgall2 = dffloat_gr
    volmax = df_vol[dd_opval].max()
    countmax = df_vol['count'].max()
    dfgall2[dd_volspec] = df_vol[dd_opval].round(4)
    grname = dfgall2.index.values
    dfgall2.shape
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(dfgall2)
    dfgall2.loc[:, :] = scaled_values
    dfgall2 = dfgall2.reset_index(drop=True)

    # Getting coordinates XYZ from Float dataframes
    x, y = np.meshgrid(range(0, len(dfgall2.columns)), dfgall2.index)
    z = dfgall2.values
    sizep = df_vol[dd_opval].values
    count = df_vol['count'].values
    z2 = dffloat.values
    xn = []
    yn = []
    zn = []
    zn2 = []
    sizepn = []
    countn = []
    xyz = []
    grnamen = []
    for i, el in enumerate(z):
        zn.append([])
        zn2.append([])
        yn.append([])
        xn.append([])

        for i2, val in enumerate(el):
            if 1 == 1:
                zn[i].append(val)
                zn2[i].append(z2[i][i2])
                yn[i].append(y[i][i2])
                xn[i].append(x[i][i2])
        sizepn.append(sizep[i])
        countn.append(count[i])
        grnamen.append(grname[i])
    znf = []
    ynf = []
    xnf = []
    zn2f = []
    sizepnf = []
    countnf = []
    grnamenf = []
    for el in zn:
        for ue in el:
            znf.append(ue)
    for el in zn2:
        for ue in el:
            zn2f.append(ue)
    for el in xn:
        for ue in el:
            xnf.append(ue)
    for i, el in enumerate(yn):
        for val in el:
            ynf.append(val)
            sizepnf.append(sizepn[i])
            countnf.append(countn[i])
            grnamenf.append(grnamen[i])
    countnf = [1 if x == 0 else x for x in countnf]
    my_xticks = dfgall2.columns
    dfgarray = dfgall2.to_numpy()
    dfgarray = dfgarray.flatten()
    arrayall = np.array([znf, zn2f,  ynf, xnf, sizepnf, countnf, dfgarray])

    # Form of Dataframe for float values
    arraynames = np.array([grnamenf])
    arrayall = np.transpose(arrayall)
    arraynames = np.transpose(arraynames)
    dfall2 = pd.DataFrame(data=arrayall, columns=[
                          'z', 'z2', 'y', 'x', 'The total volume', 'Quantity of elements', 'Value'], index=arraynames)
    dfall2['y2'] = dfall2.index
    dfnames = pd.DataFrame(data=arraynames, columns=['names'])
    multsizepnf = []
    for el in sizepnf:
        if el == 0:
            multsizepnf.append(el*volmax)
        else:
            multsizepnf.append(el*volmax)
    countnf = [element * countmax for element in countnf]
    a = np.array(dffloat_gr.columns)
    dfall2['x2'] = np.resize(a, dfall2.shape[0])
    dfall2['Typ'] = dc_forms2

    # Data connection for categorical and text values
    dfall = pd.concat([dfall, dfall2])
    dfall = dfall.sort_values(by=['y2'])
    dfall['y2'] = dfall['y2'].astype(str)
    dfall['y2'] = dfall['y2'].replace(
        to_replace=r'\'\,\)', value='', regex=True)
    dfall['y2'] = dfall['y2'].replace(
        to_replace=r'^\(\'', value='', regex=True)
    dfall['avvol'] = round(dfall['The total volume'].astype(
        float)/dfall['Quantity of elements'].astype(float), 1)

    # Additional scaling und clean data for color and size
    dfall['z'] = dfall['z'].astype(float)
    dfall = dfall[dfall.z > 0]
    dfall['Total volume'] = minmax_scale(dfall['The total volume'])
    dfall['Amount of elements'] = minmax_scale(dfall['Quantity of elements'])
    d = dfall[dfall['Amount of elements'] > 0]
    dfgf = dfall[dfall['Amount of elements'] > 0]
    conmin = round(dfgf['Amount of elements'].min(), 4)
    d = dfall[dfall['Total volume'] > 0]
    dfgf = dfall[dfall['Total volume'] > 0]
    volmin = round(dfgf['Total volume'].min(), 4)
    dfall['Amount of elements'] = dfall['Amount of elements'].replace(
        0, conmin)
    dfall['Total volume'] = dfall['Total volume'].replace(0, volmin)
    dfall.loc[dfall['Amount of elements'] == conmin, 'Typ'] = 'circle-open'
    dfall.z2 = dfall.z2.fillna(0)
    dfall = dfall[dfall.z2 != 0]
    dfall = dfall[dfall.z2 != "None"]
    dfall = dfall[dfall.z2 != np.nan]
    leng = len(dfall['x'].values)
    lengsm = round(leng/5, 0)
    numm = []
    numm.append(1)
    while i < (leng-1):
        numm.append(int(i+lengsm))
        i = i+lengsm
    dfallticks = dfall.reset_index()
    dfallticks = dfallticks.loc[numm[:-1]]
    return dfall, dfallticks

# Uploading the CSV file to the site


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    df = []
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df, html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
    ])


# Default values for filters
sizcolorall = ['Total volume', 'Amount of elements']
sizcolorall2 = ['Total volume', 'Amount of elements']
sizeminx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
allforms = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
volnamedr = ['Volume', 'Area', 'Length']
allop = ['sum', 'mean', 'max', 'median', 'min', 'first', 'last']
fig = []

# Options for displaying graphs
bgcolor = "#f3f3f1"  # mapbox light map land color
bar_bgcolor = "#b0bec5"  # material blue-gray 200
bar_unselected_color = "#78909c"  # material blue-gray 400
bar_color = "#546e7a"  # material blue-gray 700
bar_selected_color = "#37474f"  # material blue-gray 800
bar_unselected_opacity = 0.8

# Figure template
row_heights = [150, 500, 300]
template = {"layout": {"paper_bgcolor": bgcolor, "plot_bgcolor": bgcolor}}


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])
available_indicators = []
app.title = 'File data visualization'

# App HTML Layout
app.layout = html.Div([
    html.Div(
        children=[
            html.A(
                html.Div(
                    children=html.Img(
                        src="https://opendatabim.com/wp-content/uploads/2021/11/VENDOR-FREE-5.png",
                        style={"display": "inline-block", "float": "left", "height": "55px",
                               "padding": "-6px", "margin-top": "-3px", "margin-left": "40px"}
                    ),
                ),
                href="https://opendatabim.com/#", target="_blank",),
            html.H2(children="File data visualization Revit, IFC through a CSV file", style={
                'margin-left': '45px',  'margin-top': '10px', 'font-family': 'Roboto',  'display': 'inline-block', 'font-weight': '500',  'font-size': '25px'}),
            html.A(
                id='gh-link',
                children=[
                    'View on GitHub'
                ],
                href="http://github.com/#",
                style={'color': 'white',
                       'border': 'solid 1px white',
                       'text-decoration': 'none',
                       'font-size': '10pt',
                       'font-family': 'sans-serif',
                       'color': '#fff',
                       'border': 'solid 1px #fff',
                       'border-radius': '2px',
                                        'padding': '2px',
                                        'padding-top': '5px',
                                        'padding-left': '15px',
                                        'padding-right': '15px',
                                        'font-weight': '100',
                                        'position': 'relative',
                                        'top': '15px',
                                        'float': 'right',
                                        'margin-right': '40px',
                                        'margin-left': '5px',
                                        'transition-duration': '400ms',
                       }
            ),
            html.Div(
                className="div-logo",
                children=html.Img(
                    className="logo", src=("https://opendatabim.io/wp-content/uploads/2021/12/GitHub-Mark-Light-64px-1.png"),
                    style={'height': '48px',
                           'padding': '6px', 'margin-top': '3px'}
                ), style={'display': 'inline-block', 'float': 'right'}
            ),
        ], style={"background": "#2c5c97", "color": "white", "padding-top": "15px", "padding-left": "48px", "padding-bottom": "25px", "padding-left": "24px"}
    ),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H2(
                        children=[
                            html.A(
                                html.Img(
                                    src="https://opendatabim.com/wp-content/uploads/2021/11/Ein-bisschen-Text-hinzufugen-5-1.png",
                                    style={"float": "left", "width": "100%",
                                           'vertical-align': 'middle', "padding-bottom": "30px"},
                                ),
                                href="https://opendatabim.com/",
                            ),
                        ],
                        style={"text-align": "left", },
                    ),
                ], style={'width': '50%', 'display': 'inline-block', }),
                html.Div([
                    html.Div(
                        children=[
                            "Any medium-sized construction project is a source of big data with hundreds of thousands of different elements, which in turn have tens or hundreds of different parameters or properties. Up from now to properly understand this data you can use the BIMJSON format to visualize all the information on all elements and its properties as a multidimensional point cloud.",
                        ],
                        style={"text-align": "left", 'font-size': '13px', },
                    ),
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'bottom', 'padding-bottom': '35px'}),
            ], style={'width': '70%', "margin-left": "20px", 'display': 'inline-block', "margin-top": "30px", }),
            html.Div([
                html.H1(
                    children=[
                        " ", " ",
                        html.A(
                            html.Img(
                                src="https://opendatabim.com/wp-content/uploads/2021/09/VENDOR-FREE-3.png",
                                style={"float": "right", "width": "90%",
                                       'vertical-align': 'top', "padding-bottom": "10px"},
                            ),
                            href="https://opendatabim.com/",
                        ),
                    ], style={"text-align": "left", },
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'bottom', 'padding-bottom': '50px'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.P(children='Uploading a CSV file', style={
                                    'margin-left':  '10px', }),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        '📥 Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                            ]),
                        ], style={'width': '55%', 'display': 'inline-block'}),
                        html.Div([
                            html.P(children='Use a ready-made dataset', style={
                                'margin-left':  '10px', 'padding-top':  '-30px', }),
                            dcc.Dropdown(
                                id='hf-dropdown',
                                options=[
                                    {'label': 'Upload files to the site',
                                        'value': 'UF'},
                                    {'label': 'Preloaded dataset House 1',
                                              'value': 'H1'},
                                    {'label': 'Preloaded dataset House 2',
                                              'value': 'H2'}
                                ],
                                value='UF',
                                style={'height': '40px',
                                       'width': '250px',

                                       'margin-top':  '7px',
                                       'margin-bottom':  '20px',
                                       'font-size': '16px'}
                            ),
                        ], style={'width': '45%',  'display': 'inline-block', "padding-left": "30px", 'vertical-align': 'top', "padding-top": "10px"}),
                    ], style={'width': '100%',   'display': 'inline-block', 'background': 'rgb(233 238 246)',
                              'border': '2px', 'border-radius': '10px', }),
                ], style={'width': '95%', 'display': 'inline-block', "margin-top": "10px", }),
            ], style={'width': '53%', 'background': 'rgb(233 238 246)', "padding-left": "40px",
                      "padding-right": "40px", 'display': 'inline-block', "padding-top": "10px", "padding-bottom": "10px", 'border': '2px', 'border-radius': '10px', }),
            html.Div([
                html.P(children='🗃 Selecting a Category and Option to Group Project Items:', style={
                       "padding-top": "15px"}),
                html.Div([
                    html.Div(id="containerc",
                             children=dcc.Checklist(
                                 id="dd_cat",
                                 options=[
                                     {"label": "Select All Regions", "value": "All"}],
                                 value=[],
                             ),
                             ),
                    html.H6(
                        children='Select all categories or one specific',
                        style={'font-size': '10px',
                               "padding-left": "15px", "padding-top": "5px"}
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Div(id="containerb",
                             children=dcc.Checklist(
                                 id="dd_groupval",
                                 options=[
                                     {"label": "Select All Regions", "value": "All"}],
                                 value=[],
                             ),
                             ),
                    html.H6(
                        children='Parameter by which to group elements',
                        style={'font-size': '10px',  "padding-left": "15px", "padding-top": "5px"}),
                ], style={'width': '45%',  "margin-left": "30px", "padding-top": "0px", 'display': 'inline-block'}),

            ], style={'width': '42%', "margin-left": "50px",  'display': 'inline-block', 'background': 'rgb(233 238 246)', "padding-left": "50px",
                      "padding-right": "30px", "padding-top": "-10px", "padding-bottom": "30px", 'border': '2px', 'border-radius': '10px', }),
        ]),
        html.Div([
            html.Div(id='output-data-upload'),
            html.Div([
                html.Div(id='dd-output-container', style={
                    'margin-left':  '50px',
                    'font-size': '16px'}),
                html.Div([
                    dcc.Graph(id='Main-Graph', animate=True,
                              animation_options={"frame": {"redraw": True}}),
                    dcc.Interval(
                        id='graph-update',
                        interval=1
                    ),
                ], style={'width': '45%', 'display': 'inline-block', 'height': 700, }),
                html.Div([
                    html.Div([
                        html.Div([
                            html.P("🎛 When grouping are summed up:"),
                            dcc.Dropdown(
                                id='dd_volspec',
                                options=[{'label': i, 'value': i}
                                         for i in volnamedr],
                                value='Volume'
                            ),
                            html.P(
                                children='This parameter is taken for the group',
                                style={'font-size': '10px',
                                       "padding-left": "15px", "padding-top": "5px"}
                            ),

                        ], style={'width': '45%',  'display': 'inline-block'}),
                        html.Div([
                            html.P("🧮 All parameter values:"),
                            dcc.Dropdown(
                                id='dd_opval',
                                options=[{'label': i, 'value': i}
                                         for i in allop],
                                value='sum'
                            ),
                            html.P(
                                children='The values are summed if "sum"',
                                style={'font-size': '10px',
                                       "padding-left": "15px", "padding-top": "5px"}
                            ),
                        ], style={'width': '45%', "margin-left": "40px", 'display': 'inline-block'}),
                    ],),
                    html.Div([
                        html.P("📏 The point size is indicated by:"),
                        dcc.Dropdown(
                            id='dd_size',
                            options=[{'label': i, 'value': i}
                                     for i in sizcolorall],
                            value='Total volume'
                        ),
                        html.P(
                            children='The determining factor for size',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "5px"}
                        ),
                    ], style={'width': '45%', 'display': 'inline-block', }),
                    html.Div([
                        html.P("🎨 Dot color intensity:"),
                        dcc.Dropdown(
                            id='dd_color',
                            options=[{'label': i, 'value': i}
                                     for i in sizcolorall2],
                            value='Amount of elements'
                        ),
                        html.P(
                            children='The determining factor for color',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "5px"}
                        ),
                    ], style={'width': '45%', "margin-left": "40px", "margin-top": "5px", 'display': 'inline-block', }),
                    html.Div([
                        html.P("Figure for numerical properties:"),
                        dcc.Dropdown(
                            id='dc_forms1',
                            options=[{'label': i, 'value': i}
                                     for i in allforms],
                            value='circle'
                        ),
                        html.H6(
                            children='Defining the display of a property',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "5px"}
                        ),
                    ], style={'width': '45%', 'display': 'inline-block', }),
                    html.Div([
                        html.P("Figure for numerical properties:"),
                        dcc.Dropdown(
                            id='dc_forms2',
                            options=[{'label': i, 'value': i}
                                     for i in allforms],
                            value='diamond'
                        ),
                        html.H6(
                            children='Defining the display of a property',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "5px"}
                        ),
                    ], style={'width': '45%', "margin-left": "40px", 'display': 'inline-block', "margin-top": "5px"}),
                    html.Div([
                        html.P("Minimum size:"),
                        dcc.Slider(
                            id='dd_sizemin',
                            min=0.1,
                            max=8,
                            step=0.2,
                            value=1
                        ),
                        html.H6(
                            children='Minimum group size',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "0px"}
                        ),
                    ], style={'width': '45%', 'display': 'inline-block', "margin-top": "30px", }),
                    html.Div([
                        html.P("Maximum size:"),
                        dcc.Slider(
                            id='dd_sizemax',
                            min=0.5,
                            max=7,
                            step=0.05,
                            value=2
                        ),
                        html.H6(
                            children='Maximum group size',
                            style={'font-size': '10px',
                                   "padding-left": "15px", "padding-top": "0px"}
                        ),
                    ], style={'width': '45%', "margin-left": "40px", 'display': 'inline-block', "margin-top": "30px"}),
                ], style={'width': '43%', 'float': 'right', 'display': 'inline-block', 'background': bgcolor, "padding": "50px",  'border': '2px', 'border-radius': '10px', })
            ], style={'height': 700}),
        ],  style={"margin-left": "40px", "margin-top": "20px", }),
    ], style={"margin-right": "100px", "margin-left": "100px", }),
], )

# Upload the file to the application and create a category filter
@app.callback([
            Output('output-data-upload', 'children'), 
            Output("containerc", "children"), 
            Output("containerb", "children"), ],
            Input('hf-dropdown', 'value'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified')
              )
def update_output(valuedd, list_of_contents, list_of_names, list_of_dates):
    children = []
    retx = []
    file = '/var/www/qto/data/1house.csv'
    df = pd.read_csv(file, low_memory=False, nrows=10)

    # Retrieving data from a CSV file
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(
            list_of_contents, list_of_names, list_of_dates)]
        df = children[0][0]
        retx = children[0][1]

    # If dropdown menu is selected, select file from server
    if valuedd == 'H1':
        print('H1...........')
        file = '/var/www/qto/data/1house.csv'
        retx = '1house.csv'
        df = pd.read_csv(file, low_memory=False, nrows=10)
    elif valuedd == 'H2':
        file = '/var/www/qto/data/6house.csv'
        df = pd.read_csv(file, low_memory=False, nrows=10)
        retx = '2house.csv'
    else:
        pass

    # Define a category for filter selection
    dfi = df
    onlycat = dfi['Category'].unique()
    dfi['Category'].unique()
    onlycat = np.insert(onlycat, 0, 'All categories')
    allpropdf = dfi.columns
    return [retx,
            dcc.Dropdown(
                id='dd_cat',
                options=[{'label': i, 'value': i} for i in onlycat],
                value='All categories'
            ),
            dcc.Dropdown(
                id='dd_groupval',
                options=[{'label': i, 'value': i} for i in allpropdf],
                value='Type'
            ), ]

# Basic Callback for displaying a chart by parameters
@app.callback(
    Output("Main-Graph", "figure"),
    [
        Input('hf-dropdown', 'value'),
        Input("dd_cat", "value"),
        Input("dd_groupval", "value"),
        Input("dd_volspec", "value"),
        Input("dd_opval", "value"),
        Input("dd_size", "value"),
        Input("dd_color", "value"),
        Input("dd_sizemin", "value"),
        Input("dd_sizemax", "value"),
        Input("dc_forms1", "value"),
        Input("dc_forms2", "value"),
        State('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified')
    ])
def update_figure2(valuedd, dd_cat, dd_groupval, dd_volspec, dd_opval, dd_size, dd_color, dd_sizemin, dd_sizemax, dc_forms1, dc_forms2, list_of_contents, list_of_names, list_of_dates):

    # Selecting data from a loaded file or from preloaded projects
    file = '/var/www/qto/data/1house.csv'
    df = pd.read_csv(file, low_memory=False)
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(
            list_of_contents, list_of_names, list_of_dates)]
        df = children[0][0]
    if valuedd == 'H1':
        file = '/var/www/qto/data/1house.csv'
        df = pd.read_csv(file, low_memory=False)
    elif valuedd == 'H2':
        file = '/var/www/qto/data/6house.csv'
        df = pd.read_csv(file, low_memory=False)
    else:
        pass
    dfi = df
    onlycat = dfi['Category'].unique()
    dfi['Category'].unique()
    onlycat = np.insert(onlycat, 0, 'All categories')
    allpropdf = dfi.columns

    # Function for displaying the graph
    def update_figure():
        dfc, df = f_cat(dd_cat, dfi)
        dfall, dfallticks = cr_df_floatstring(
            dd_volspec, dd_groupval, dd_opval, dc_forms1, dc_forms2, dfc, df)
        vol = dfall["The total volume"]
        fig = go.Figure(data=go.Scatter3d(
            x=dfall['x'],
            y=dfall['z'],
            z=dfall['y'],
            hoverinfo="text",
            text=[f"Type: {a}<br>The total volume: {b} m³<br>Amount: {c}<br>Avr. volume: {e} m³ <br>Property: {d}<br>Value: {f}" for a, b, e, c, d, f in list(
                zip(dfall['y2'], dfall['The total volume'], dfall['avvol'], dfall['Quantity of elements'], dfall['x2'], dfall['z2']))],
            mode='markers',
            marker=dict(
                sizemode='area',
                sizeref=0.001/dd_sizemax,
                sizemin=10*dd_sizemin,
                symbol=dfall['Typ'],
                size=dfall[dd_color],
                opacity=0.5,
                color=dfall[dd_size],
                colorscale='jet',
                line_color='rgb(140, 140, 170)'
            ),
        ))

        fig.update_layout(scene=dict(
            yaxis=dict(
                tickfont=dict(
                    size=10,
                    family='Helvetica',
                ),
                nticks=0,
                ticktext=[],
                tickvals=[],
            ),
            xaxis=dict(
                tickfont=dict(
                    size=10,
                    family='Helvetica',
                ),
                ticktext=dfallticks['x2'].values,
                tickvals=dfallticks['x'].values,
                #align = 'left',
                ticks='outside',
                # ticks='outside',
                nticks=4,
            ),
            zaxis=dict(
                tickfont=dict(
                    size=10,
                    family='Helvetica',
                ),
                ticktext=dfallticks['y2'].values,
                tickvals=dfallticks['y'].values,
                ticks='outside',
            ),
        ),
            width=700,
            margin=dict(r=0, l=10, b=10, t=0)
        )

        fig.update_layout(template="seaborn")

        fig.update_layout(scene=dict(
            xaxis_title='Properties',
            yaxis_title='Property value',
            zaxis_title='Groups of elements'),
            font=dict(
            size=20,
            family='Helvetica',
        ),
            width=700,
            height=700,
            margin=dict(r=0, b=10, l=10, t=0))
        name = 'eye = (x:2, y:2, z:2)'
        camera = dict(
            eye=dict(x=1.25, y=1.25, z=1.5)
        )

        fig.update_traces(selector=dict(type='heatmap'))

        fig.update_scenes(yaxis_autorange="reversed")
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig
    fig = update_figure()
    return fig
   

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=3000, use_reloader=True,)

