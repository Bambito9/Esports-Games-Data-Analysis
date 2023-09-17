import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import vtk
from vtk import vtkStructuredPoints
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'false'

df = pd.read_csv("/home/bambito9/Esports-Data-Analysis/API Requests/League of Legends/Datasets/Professional/FinalProfessionalPlayersDataset.csv", dtype = {'skillSlot': str, 'buildingType': str, 'lane': str, 'monster': str, "itemName": str})
#df

jungler_deaths_df = df[(df["role"] == "JUNGLE") & (df["eventType"] == "Death")]
#jungler_deaths_df

redSide_jungler_deaths_df = jungler_deaths_df[jungler_deaths_df["team"] == "Red"]
#redSide_jungler_deaths_df

redSide_jungler_deaths_df = redSide_jungler_deaths_df[["coordinate_x","coordinate_y","timestamp"]].copy()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(redSide_jungler_deaths_df)
data_scaled

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_scaled)

redSide_jungler_deaths_df["cluster"] = kmeans.labels_

pv.start_xvfb()
points = np.column_stack((redSide_jungler_deaths_df['coordinate_x'], redSide_jungler_deaths_df['coordinate_y'], redSide_jungler_deaths_df['timestamp']))
grid = pv.UnstructuredGrid()
grid.points = points

cluster_labels = kmeans.labels_
print(type(cluster_labels))
cluster_labels = np.expand_dims(cluster_labels, axis=1)
print(cluster_labels)
cluster_data = vtk.vtkIntArray()
cluster_data.SetNumberOfComponents(1)
cluster_data.SetName('cluster')
cluster_data.SetArray(cluster_labels, grid.GetNumberOfPoints(), 1)

grid.GetPointData().AddArray(cluster_data)

print(grid.GetPoints())
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='cluster', cmap='Set1')

#plotter.export_vtkjs("my_plot.vtkjs")

plotter.camera_position = [(redSide_jungler_deaths_df['coordinate_x'].min()+redSide_jungler_deaths_df['coordinate_y'].max())/2, (redSide_jungler_deaths_df['coordinate_x'].min()+redSide_jungler_deaths_df['coordinate_y'].max())/2, redSide_jungler_deaths_df['timestamp'].max()*2]
plotter.camera_orientation = [180, -90, 0]

plotter.enable_anti_aliasing()

plotter.show()

#screenshot = plotter.screenshot()

#Image.SAVE("plot.png")