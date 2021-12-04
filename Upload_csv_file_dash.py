import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
fig = go.Figure()
import dash_table
import plotly.express as px
import dash_daq as daq
import pandas as pd
from classification import *
from regression import *
import pydot



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Start the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
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
    html.Div(id='output-div'),
    html.Div(id='output-datatable'),
    html.Div(id='output-division'),
    html.Div(id='output-division2'),
    html.Div(id='output-division3'),
    html.Div(id='output-division4'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

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

    return html.Div([
        html.H5("Loaded file : " + filename),
        html.H6('View the data', style={"font-weight": "bold"}),
        html.P("Choose graph", style={"color": "grey"}),
        dcc.RadioItems(id="graph-selected",
                       options=[{'label': 'Bar Graph', 'value':'bar'},
                                {'label': 'Scatter Plot', 'value':'scatter'}],
                       value='bar'
        ),
        html.P("Inset X axis data", style={"color": "grey"}),
        dcc.Dropdown(id='xaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        html.P("Inset Y axis data", style={"color": "grey"}),
        dcc.Dropdown(id='yaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns]),
        
        # Button for visualize the data  
        html.Button(id="submit-button", children="Create graph",style={
             'background-color': '#2dbecd',
              'color': 'white'}),
        html.Hr(),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        
        # Horizontal line
        html.Hr(),  
        
        
        html.H6('Choose the data',style={"font-weight": "bold"}),
        html.P("Target variable", style={"color": "grey"}),
        dcc.Dropdown(id='target',
                      options=[{'label':x, 'value':x} for x in df.columns]),
        html.P("Predictor variables", style={"color": "grey"}),
        dcc.Dropdown(id='predictor',
                      options=[{'label':x, 'value':x} for x in df.columns], multi=True),
        html.Button(id="submit-button2", children="Show algorithm",style={
             'background-color': '#2dbecd',
              'color': 'white'}),
        html.Br(),
        html.Br()
        
    ])

# DataTable output
@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# Visualization output
@app.callback(Output('output-div', 'children'),
              Input('submit-button','n_clicks'), # Show the graph only after the click on the button
              State('graph-selected','value'),
              State('stored-data','data'),
              State('xaxis-data','value'),
              State('yaxis-data', 'value'))
def make_graphs(n, graph_chosen, data, x_data, y_data):
    if n is None:
        return dash.no_update
    elif graph_chosen == 'bar':
        bar_fig = px.bar(data, x=x_data, y=y_data)
        return dcc.Graph(figure=bar_fig)
    elif graph_chosen == 'scatter':
        scatter_fig = px.scatter(data, x=x_data, y=y_data)
        return dcc.Graph(figure=scatter_fig)
    
# Get the type of the target variable
@app.callback(Output('output-division', 'children'),
              Input('submit-button2','n_clicks'),
              Input('stored-data','data'),
              Input('target','value'))
def algo_proposed(n, data, target):
    data = pd.DataFrame(data)
    if n is None:
        return dash.no_update
    elif data[target].dtypes != "object":
        return html.Div([
            html.P("Select your regression algorithm", style={"color": "grey"}),
            dcc.Dropdown(id='regress',
                      options=[{'label':'Linear Regression', 'value':'linear'},
                                {'label':'Regression tree', 'value':'regtree'},
                                {'label':'Support Vector Regression', 'value':'vectreg'}]),
            ])
    else:
        return html.Div([
            html.P("Select your classification algorithm", style={"color": "grey"}),
            dcc.Dropdown(id='classif',
                      options=[{'label':'Decision tree', 'value':'tree'},
                                {'label':'Discriminant analysis', 'value':'adl'},
                                {'label':'Regression logistic', 'value':'reglog'}]),
        ])


@app.callback(Output('output-division2', 'children'),
              Input('submit-button2','n_clicks'),
              Input('target','value'))    
def start_algo(n,target):
    return html.Div([
        
        # Horizontal line
        html.Hr(),  
        
        html.H6('Predict the data',style={"font-weight": "bold"}),
        html.Button(id="submit-button3", children="Start algorithm",style={
             'background-color': '#2dbecd',
              'color': 'white'})
    ])

 
@app.callback(Output('output-division3', 'children'),
              Input('submit-button3','n_clicks'),
              Input('classif','value'),
              Input('stored-data','data'),
              Input('target','value'),
              Input('predictor', 'value'))
def start_algo_classif(n, classif, data, target, predictor):
    data = pd.DataFrame(data)
    
    if n is None:
        return dash.no_update
    else:
        if classif == "tree":
            best_param, best_score, acc, matrix_confusion, time_execution, fig_ROC, table_param, tree = classification.arbre_de_decision(data[target],data[predictor])
            return html.Div(
                    [
                          html.Br(),
                          html.H5("Best parameter", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                          dcc.Graph(figure=table_param),
                          html.Br(),
                          html.Div(
                              [
                                  html.Br(),
                                  daq.LEDDisplay(
                                      id='best_score',
                                      value=best_score,
                                      label = "Best score",
                                      size=20,
                                      color = '#77079D'
                                  ),html.Br(),
                                  daq.LEDDisplay(
                                      id='time_execution',
                                      value=time_execution,
                                      label = "Time of execution",
                                      size=20,
                                      color = '#77079D'
                                  ),html.Br(),
                                  daq.LEDDisplay(
                                      id='acc',
                                      value=acc,
                                      label = "Accuracy",
                                      size=20,
                                      color = '#77079D'
                                  ),
                              ],className='pretty_container two columns',
                        ),
                        html.H5("Confusion Matrix", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        dcc.Graph(figure = matrix_confusion, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Div([
                                html.H5("ROC curve", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                dcc.Graph(figure=fig_ROC, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                html.H5("Decision tree explanation", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                                dcc.Graph(figure=tree, style={'width': '90%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'})
                            ])
                        
                        ])
        
        if classif == "reglog":
            best_param, best_score, acc, matrix_confusion, time_execution, fig_ROC, table_param = classification.reg_log(data[target],data[predictor])
            
        if classif == "adl":
            best_param, best_score, acc, matrix_confusion, time_execution, fig_ROC, table_param = classification.adl(data[target],data[predictor])
            
        return html.Div(
                [
                      html.Br(),
                      html.Div(
                          [
                              html.H5("Best parameter", style={'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                              dcc.Graph(figure=table_param),
                              html.Br(),
                              daq.LEDDisplay(
                                  id='best_score',
                                  value=best_score,
                                  label = "Best score",
                                  size=20,
                                  color = '#77079D'
                              ),html.Br(),
                              daq.LEDDisplay(
                                  id='time_execution',
                                  value=time_execution,
                                  label = "Time of execution",
                                  size=20,
                                  color = '#77079D'
                              ),html.Br(),
                              daq.LEDDisplay(
                                  id='acc',
                                  value=acc,
                                  label = "Accuracy",
                                  size=20,
                                  color = '#77079D'
                              ),
                          ],className='pretty_container two columns',
                    ),
                    html.H5("Confusion Matrix", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    dcc.Graph(figure = matrix_confusion, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    html.H5("ROC curve", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    dcc.Graph(figure=fig_ROC, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'})
                    ])
    
                    
                   
    
@app.callback(Output('output-division4', 'children'),
              Input('submit-button3','n_clicks'),
              Input('regress','value'),
              Input('stored-data','data'),
              Input('target','value'),
              Input('predictor', 'value'))
def start_algo_regress(n, regress, data, target, predictor):
    data = pd.DataFrame(data)
    
    if n is None:
        return dash.no_update
    else:
        if regress == "linear":
            best_param, best_score, r_carre, mse, time_execution, scatter_fig, table_param = regression.linear_reg(data[target],data[predictor])
        
        if regress == "regtree":
            best_param, best_score, r_carre, mse, time_execution, scatter_fig, table_param = regression.dec_tree_reg(data[target],data[predictor])
        
        if regress == "vectreg":
            best_param, best_score, r_carre, mse, time_execution, scatter_fig, table_param = regression.svr(data[target],data[predictor])
        
        
        return html.Div(
                [
                      html.Div(
                          [
                              html.Br(),
                              html.H5("Best parameter", style={'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                              dcc.Graph(figure=table_param, style={'width': '25%'}),
                              html.Br(),
                              html.Div(
                                  [
                                      daq.LEDDisplay(
                                          id='best_score',
                                          value=best_score,
                                          label = "Best score",
                                          size=20,
                                          color = '#77079D'
                                      ),html.Br(),
                                      daq.LEDDisplay(
                                          id='mse',
                                          value=mse,
                                          label = "MSE",
                                          size=20,
                                          color = '#77079D'
                                      ),html.Br(),
                                  ],className='pretty_container two columns',
                        ),
                    ]),
                    html.Div(
                        [
                            html.Div(
                                [
                                    daq.LEDDisplay(
                                        id='r_carre',
                                        value=r_carre,
                                        label = "R-squared",
                                        size=20,
                                        color = '#77079D'
                                    ),html.Br(),
                                    daq.LEDDisplay(
                                        id='time_execution',
                                        value=time_execution,
                                        label = "Time of execution",
                                        size=20,
                                        color = '#77079D'
                                    ),html.Br(),
                                ],className='pretty_container two columns',
                            )
                        ]),
                    html.H5("Scatter plot", style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    dcc.Graph(figure=scatter_fig, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'})
                    ])
        
  
    
app.run_server(port=8040, use_reloader=False)

    
    
    
    
    