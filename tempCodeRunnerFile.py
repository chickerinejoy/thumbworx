# app.py
from flask import Flask, request, jsonify
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
import logging
from haversine import haversine
import folium
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(filename="activity.log", level=logging.INFO)
def log_activity(msg): logging.info(f"{datetime.utcnow()}: {msg}")

# -----------------------------
# Real-time data stores
# -----------------------------
drivers = [
    {"id":1,"imei":"123456789012345","driver":"Alice","lat":14.5547,"lon":121.0244,"current_load":1},
    {"id":2,"imei":"234567890123456","driver":"Bob","lat":14.5550,"lon":121.0300,"current_load":2},
    {"id":3,"imei":"345678901234567","driver":"Charlie","lat":14.5600,"lon":121.0200,"current_load":0},
]

deliveries = []  # will be added in real time

geofences = [
    {"id":1,"name":"No-Go Zone Makati","polygon":[(14.555,121.023),(14.556,121.023),(14.556,121.025),(14.555,121.025)]}
]

# -----------------------------
# Load Makati City Graph
# -----------------------------
print("Loading city graph...")
G = ox.graph_from_place("Makati City, Philippines", network_type="drive")

# -----------------------------
# Helper functions
# -----------------------------
def suggest_route(origin, destination):
    try:
        orig_node = ox.nearest_nodes(G, origin[1], origin[0])
        dest_node = ox.nearest_nodes(G, destination[1], destination[0])
        nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in nodes]
        distance_m = sum(nx.shortest_path_length(G, orig_node, dest_node, weight='length') for n in nodes)
        eta_min = distance_m / 500  # assume 30 km/h ~ 500 m/min
        return coords, round(eta_min, 1)
    except:
        return [], 0

def assign_driver(delivery):
    best_driver = min(
        drivers,
        key=lambda d: haversine((d['lat'], d['lon']),(delivery['lat'], delivery['lon'])) + d['current_load']
    )
    delivery['assigned_driver'] = best_driver['id']
    best_driver['current_load'] += 1
    log_activity(f"Assigned delivery {delivery['id']} to driver {best_driver['driver']}")
    return best_driver

def cluster_deliveries(deliveries, n_clusters=2):
    if len(deliveries) <= 1: return {0: deliveries}
    coords = np.array([[d['lat'], d['lon']] for d in deliveries])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(deliveries[idx])
    return clusters

def check_geofence(lat, lon):
    point = Point(lat, lon)
    for gf in geofences:
        poly = Polygon(gf['polygon'])
        if poly.contains(point):
            return gf['name']
    return None

def plot_map():
    m = folium.Map(location=[14.5547,121.0244], zoom_start=13)
    # Drivers
    for d in drivers:
        folium.Marker([d['lat'], d['lon']], popup=d['driver'], icon=folium.Icon(color='blue')).add_to(m)
    # Deliveries
    for deliv in deliveries:
        color = 'green' if deliv.get('assigned_driver') else 'red'
        folium.Marker([deliv['lat'], deliv['lon']], popup=f"Delivery {deliv['id']}", icon=folium.Icon(color=color)).add_to(m)
    # Geofences
    for gf in geofences:
        folium.Polygon(gf['polygon'], color='red', fill=True, fill_opacity=0.3, popup=gf['name']).add_to(m)
    return m._repr_html_()

# -----------------------------
# API Endpoints
# -----------------------------

# 1. Add delivery request
@app.route("/delivery/request", methods=["POST"])
def add_delivery():
    data = request.json
    delivery = {
        "id": len(deliveries)+1,
        "lat": data['lat'],
        "lon": data['lon'],
        "address": data.get('address','Unknown'),
        "assigned_driver": None,
        "requested_time": datetime.utcnow()
    }
    deliveries.append(delivery)
    log_activity(f"New delivery request {delivery['id']} at {delivery['lat']},{delivery['lon']}")
    return jsonify({"status":"success","delivery":delivery})

# 2. Assign deliveries to drivers + recommend routes
@app.route("/delivery/assign", methods=["GET"])
def assign_all():
    response = []
    routes = []
    for d in deliveries:
        if d['assigned_driver']: continue
        if check_geofence(d['lat'], d['lon']):
            log_activity(f"Delivery {d['id']} inside geofence, skipping")
            continue
        driver = assign_driver(d)
        route_coords, eta = suggest_route((driver['lat'], driver['lon']), (d['lat'], d['lon']))
        d['eta_min'] = eta
        routes.append(route_coords)
        response.append({
            "delivery_id": d['id'],
            "driver": driver['driver'],
            "route": route_coords,
            "eta_min": eta
        })
    map_html = plot_map()
    return jsonify({"assignments": response, "map_html": map_html})

# 3. Map view
@app.route("/map", methods=["GET"])
def get_map():
    return plot_map()

# 4. Geofence check
@app.route("/geofence/check", methods=["POST"])
def process_geofence():
    data = request.json
    gf_name = check_geofence(data['lat'], data['lon'])
    return jsonify({"inside_geofence": gf_name is not None, "geofence_name": gf_name})

# 5. Activity logs
@app.route("/activity_logs", methods=["GET"])
def get_logs():
    with open("activity.log") as f:
        logs = f.readlines()
    return jsonify({"logs": logs[-50:]})

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
