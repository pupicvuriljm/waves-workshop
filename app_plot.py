## Interactive lag plotting app using Dash

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# -------------------------
# Load data
# -------------------------
lags = pd.read_parquet("lags.parquet")
station_info = pd.read_csv("stations_info.csv")

stations = lags.index.tolist()
lat = station_info.set_index("filename").loc[stations, "latitude"].to_numpy()
lon = station_info.set_index("filename").loc[stations, "longitude"].to_numpy()
points = np.column_stack((lon, lat))

# -------------------------
# Interpolation grid (coarser for performance)
# -------------------------
xi = np.linspace(min(lon) - 1, max(lon) + 2, 200)
yi = np.linspace(min(lat) - 1, max(lat) + 2, 200)
Xi, Yi = np.meshgrid(xi, yi)

# -------------------------
# Discrete levels
# -------------------------
levels = np.arange(-12, 22, 3)
colorscale = "Magma"

# -------------------------
# Initialize Dash app
# -------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interpolated surge series lag North Sea", style={'textAlign': 'center'}),
    
    # Dropdown above the graph
    html.Div(
        style={'width': '40%', 'padding': '10px'},  # centered
        children=[
            dcc.Dropdown(
                id='station-dropdown',
                options=[{'label': station_info.set_index('filename').loc[s, 'name'], 'value': s} for s in stations],
                value=stations[0]
            )
        ]
    ),
    # Graph below
    html.Div(
        children=[
            dcc.Graph(id='lag-map', style={'height': '90vh', 'width': '95%', 'margin': '0 auto'})
        ]
    )
])

# -------------------------
# Callback to update figure
# -------------------------
@app.callback(
    Output('lag-map', 'figure'),
    Input('station-dropdown', 'value')
)
def update_figure(ref_station):
    z = lags[ref_station].to_numpy()
    Zi = griddata(points, z, (Xi, Yi), method="cubic")

    # Mask NaNs
    mask = ~np.isnan(Zi)
    Xi_masked = Xi[mask]
    Yi_masked = Yi[mask]
    Zi_masked = Zi[mask]

    Zi_discrete = np.digitize(Zi_masked, levels) - 1

    trace = go.Scattergeo(
        lon=Xi_masked,
        lat=Yi_masked,
        mode='markers',
        marker=dict(
            size=6,
            color=Zi_discrete,
            colorscale=colorscale,
            cmin=0,
            cmax=len(levels)-2,
            colorbar=dict(
                title="Lag [h]",
                tickvals=np.arange(len(levels)-1),
                ticktext=[f"{levels[i]} to {levels[i+1]}" for i in range(len(levels)-1)],
                tickmode="array",
                orientation="v",       # vertical
                len=0.7,
                x=1.02,               # slightly to the right of map
                y=0.6,
                xanchor="left",
                yanchor="middle",
                thickness=20
                )
        ),
        name='Lag',
        showlegend=False
    )

    station_trace = go.Scattergeo(
        lon=lon,
        lat=lat,
        mode='markers+text',
        text=station_info.set_index('filename').loc[stations, 'name'].tolist(),
        textposition="top center",
        marker=dict(size=10, color='black'),
        showlegend=False
    )

    fig = go.Figure(data=[trace, station_trace])
    fig.update_layout(
        title=dict(
        text=f"Lag relative to {station_info.set_index('filename').loc[ref_station, 'name']}",
        x=0,
        xanchor='left',
        y=0.93,     # (1.0 is top)
        yanchor='top'
        ),
        geo=dict(
            scope='europe',
            projection_type='mercator',
            resolution=50,
            showcountries=True,
            showland=True,
            landcolor="lightgrey",
            lataxis=dict(range=[49, 63]),
            lonaxis=dict(range=[-5, 12])
        ),
        margin=dict(l=5, r=5, t=30, b=5)
    )

    return fig

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=10000)
