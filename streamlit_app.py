"""
The Ternary Chart Program creates 1-4 ternary charts.
It takes ternary coords and outputs one plot with the chart(s).
The program was written by EK Esawi & Lily J. Meadow.
"""

import math as ma, matplotlib.pyplot as plt, numpy as np
from matplotlib.transforms import TransformedPath
from matplotlib.path import Path
import streamlit as st, pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import io

def Tlines(ax, i, verc, cc, lw, angle=0, center=(0,0), shift_x=0, shift_y=0, magnifications=1):
    """
    Draws ternary lines on a ternary plot based on specified parameters.

    Parameters:
    ax: The Matplotlib axis object.
    i: Amount of lines.
    verc: Cuts.
    cc: Colors for the lines.
    lw: Line widths for the lines.
    angle (optional, default 0): Rotation angle in degrees.
    center (optional, default (0, 0)): Center point around which to rotate.
    shift_x (optional, default 0): Shift along the x axis.
    shift_y (optional, default 0): Shift along the y axis.
    magnifications (optional, default 1): Value of magnification. 

    Returns:
    Labels for left, right, and top lines.
    """
    cval = [k[0] for k in verc]
    lval = [k[1] for k in verc]

    LS = [[100 - i, i, 0] for i in range(0, 100 + i, i)]
    RS = [[0, i, 100 - i] for i in range(0, 100 + i, i)]
    TS = [[i, 0, 100 - i] for i in range(0, 100 + i, i)]
   
    if cval[0] == "L":
        left_list1 = LS
        left_list2 = list(reversed(TS))
       
        if lval[0] > 0 and lval[0] in [k[1] for k in left_list1]:
            left_index = [k[1] for k in left_list1].index(lval[0])
            left_list1 = left_list1[left_index:]
            left_list2 = left_list2[left_index:]
       
        if cval[1] == "R" and lval[1] > 0:
            aa = [j[0] for j in left_list2].index(lval[1])
            for k in range(aa, len(left_list2)):
                left_list1[k][1] = left_list1[aa][1]
                left_list1[k][2] = 100 - sum(left_list1[k])
       
        if cval[2] == "T" and lval[2] > 0:
            bb = [j[0] for j in left_list2].index(lval[2])
            for m in range(bb, len(left_list2)):
                left_list2[m][2] = left_list2[bb][2]
                left_list2[m][1] = 100 - sum(left_list2[m])
       
        t2 = [x for pair in list(zip(left_list1, left_list2)) for x in pair]

        transformed_t2 = transform_coordinates(TtB(t2), angle, center,
                                               shift_x, shift_y, magnifications)
        Tplot(ax, transformed_t2, "dcl", cc[0], lw[0])
        ltlabs = [k[0] for k in left_list1 if k[0] >= lval[1]]

    if cval[1] == "R":
        right_list1 = list(reversed(LS))
        right_list2 = list(reversed(RS))
       
        if lval[1] > 0:
            right_list1 = right_list1[[k[0] for k in right_list1].index(lval[1]):]
            right_list2 = right_list2[[k[2] for k in right_list2].index(lval[1]):]

        if cval[0] == "L" and lval[0] > 0:
            aa = [j[1] for j in right_list1].index(lval[0])
            for k in range(aa, len(right_list1)):
                right_list1[k][0] = right_list1[aa][0]
                right_list1[k][2] = 100 - sum(right_list1[k])

        if cval[2] == "T" and lval[2] > 0:
            bb = [j[1] for j in right_list2].index(lval[2])
            for m in range(bb, len(right_list2)):
                right_list2[m][2] = right_list2[bb][2]
                right_list2[m][0] = 100 - sum(right_list2[m])
       
        t2 = [x for pair in list(zip(right_list1, right_list2)) for x in pair]

        transformed_t2 = transform_coordinates(TtB(t2), angle, center,
                                               shift_x, shift_y, magnifications)
        Tplot(ax, transformed_t2, "dcl", cc[1], lw[1])
        rtlabs = [k[1] for k in right_list1 if k[1] >= lval[2]]

    if cval[2] == "T":
        top_list1 = TS
        top_list2 = RS

        if cval[0] == "L" and lval[0] > 0:
            aa = [j[2] for j in top_list1].index(lval[0])
            for k in range(aa, len(top_list1)):
                top_list1[k][0] = top_list1[aa][0]
                top_list1[k][1] = 100 - sum(top_list1[k])
       
        if cval[1] == "R" and lval[1] > 0:
            bb = [j[2] for j in top_list2].index(lval[1])
            for m in range(bb, len(top_list2)):
                top_list2[m][1] = top_list2[bb][1]
                top_list2[m][0] = 100 - sum(top_list2[m])
       
        t2 = [x for pair in list(zip(top_list1, top_list2)) for x in pair]

        transformed_t2 = transform_coordinates(TtB(t2), angle, center,
                                               shift_x, shift_y, magnifications)
        Tplot(ax, transformed_t2, "dcl", cc[2], lw[2])
        ttlabs = [k[2] for k in top_list1 if k[2] >= lval[0]]

    return (ltlabs, rtlabs, ttlabs), (LS, RS, TS)

#============================================================================  

def TtB(tcords):
    """
    Converts ternary coordinates to Cartesian coordinates.

    Parameters:
    tcords: List of ternary coordinates

    Returns:
    List of Cartesian coordinates
    """
    tcords = [[100 * x / sum(j) for x in j] for j in tcords if sum(j) != 0]

    for i in range(len(tcords)):
       if len(tcords[0])!=3:
          print("OPS there must be 3 coordinates")
          exit()
         
       ycords=[tcords[i][2]*ma.cos(ma.radians(30)) for i in range(len(tcords))]
       xcords=[tcords[i][1]+ycords[i]*ma.tan(ma.radians(30)) for i in range(len(tcords))]
       res=list(map(list, zip(xcords,ycords)))

    return (res)

#============================================================================

def edgs(ax, cuts, angle=0, center=(0,0), shift_x=0, shift_y=0, magnifications=1, edge_color='blue'):
    """
    Draws edges on the ternary plot based on cut values.

    Parameters:
    ax: The Matplotlib axis object.
    cuts: List of cut values for each side of the ternary plot.
    angle (optional, default 0): Rotation angle in degrees.
    center (optional, default [0, 0]): Center point around which to rotate.
    shift_x (optional, default 0): Shift along the x axis.
    shift_y (optional, default 0): Shift along the y axis.
    magnifications (optional, default 1): Value of magnification. 
    edge_color (optional, default 'blue'): Color of the edge line.
    
    Returns:
    transformed_nedgs: Edge coords.
    """
   
    cuts = [k[1] for k in cuts]
    nedgs = []

    if cuts[0] > 0:
        nedgs.append([100 - cuts[0], 0, cuts[0]])
        nedgs.append([100 - cuts[0], cuts[0], 0])
    else:
        nedgs.append([100, 0, 0])

    if cuts[1] > 0:
        nedgs.append([cuts[1], 100 - cuts[1], 0])
        nedgs.append([0, 100 - cuts[1], cuts[1]])
    else:
        nedgs.append([0, 100, 0])

    if cuts[2] > 0:
        nedgs.append([0, cuts[2], 100 - cuts[2]])
        nedgs.append([cuts[2], 0, 100 - cuts[2]])
        nedgs.append(nedgs[0])
    else:
        nedgs.append([0, 0, 100])
        nedgs.append(nedgs[0])
       
    transformed_nedgs = transform_coordinates(TtB(nedgs), angle, center,
                                              shift_x, shift_y, magnifications)
    Tplot(ax, transformed_nedgs, "", edge_color, 0.8)
       
    return transformed_nedgs
   
#============================================================================

def Tplot(ax, cords, optn, cc, lw):
    """
    Plots coordinates on the ternary plot

    Parameters:
        ax: The Matplotlib axis object.
        cords: List of Cartesian coordinates.
        optn: Plotting option (e.g., "dcl").
        cc: Color.
        lw: Line width.
    """

    if optn == "dcl":
        cords = sum([[cords[j], cords[j + 1],
                [float('nan'),
                 float('nan')]] for j in range(len(cords) - 1) if j % 2 == 0],
                [])[:-1]

    xax=list(zip(*cords))[0]
    yax=list(zip(*cords))[1]

    ax.plot(xax, yax, cc, lw=lw)
   
#============================================================================

def points_in_ternary(vertices, points):
    """
    Check if points are inside the ternary plot defined by the vertices.

    Parameters:
    vertices: List of coordinates defining the vertices.
    points: List of point coordinates to check.

    Returns:
    inside.tolist(): A list indicating whether each point is inside (True) or outside (False) the plot.
    """
   
    path = Path(vertices)
   
    points_array = np.array(points)
   
    inside = path.contains_points(points_array)
   
    return inside.tolist()
    
#============================================================================        

def check_ternary_coords(coords, tolerance=1e-5):
    if abs(sum(coords) - 1) > tolerance:
        print(f"⚠️ Error: Ternary coordinates {coords} do not sum to 1. Actual sum: {sum(coords)}")
    return coords

#============================================================================

def compute_ternary_center(ax, shift_x=0, shift_y=0):
    """
    Computes the centroid of the ternary plot and plots it.

    Parameters:
    ax: The Matplotlib axis object.
    shift_x (optional, default 0): Shift along the x axis.
    shift_y (optional, default 0): Shift along the y axis.

    Returns:
    center_cartesian: The centroid in Cartesian coordinates.
    """
   
    center_ternary_initial = [(1/3)*100, (1/3)*100, (1/3)*100]
       
    center_cartesian = TtB([center_ternary_initial])[0]
   
    center_cartesian[0] += shift_x
    center_cartesian[1] += shift_y
   
    return center_cartesian

#============================================================================

def transform_coordinates(coords, angle=0, center=(0,0), shift_x=0, shift_y=0,
                          magnifications=1):
    """
    Rotate, shift, and magnify coordinates around a center.
   
    Parameters:
    coords: List of Cartesian coordinates to be transformed.
    angle (optional, default 0): Rotation angle in degrees.
    center (optional, default [0, 0]): Center point around which to rotate.
    shift_x (optional, default 0): Shift along the x axis.
    shift_y (optional, default 0): Shift along the y axis.
    magnifications (optional, default 1): Value of magnification. 
   
    Returns:
    coords: Array of the transformed coordinates.
    """
   
    coords = np.array(coords, dtype=float)
    
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)

    coords[:, 0] += shift_x
    coords[:, 1] += shift_y
   
    center_shifted = np.array(center, dtype=float)
    coords -= center_shifted
    coords *= magnifications
    coords += center_shifted

    if angle != 0:
        theta = ma.radians(angle)
       
        rotation_matrix = np.array([
            [ma.cos(theta), -ma.sin(theta)],
            [ma.sin(theta), ma.cos(theta)]
        ])
        coords[:, :2] -= center_shifted.astype(coords.dtype)
        coords[:, :2] = np.dot(coords[:, :2], rotation_matrix.T)
        coords[:, :2] += center_shifted

    return coords

#============================================================================  

def plot_on_ax(ax, data, color='blue', marker='o', label='Ternary Plot', angle=0, center=[0,0], magnifications=1,
               ticks=[["L",0],["R",0],["T",0]], colors=["red", "blue", "green"], linews=[.8, .8, .8], 
               line_starts=[], line_ends=[], seg_labels=[], shift_x=0,
               shift_y=0, num_lines=10, labels=False, contours=False,
               contour_levels=10, data_marker_size=50, cmap='viridis',
               coord_system='Ternary', segment_line_color='black', segment_line_width=1.0,
               left_label_color='red', right_label_color='blue', top_label_color='green',
               edge_color='blue', cartesian_labels=[], cartesian_label_style={},
               show_data_with_contours=True, contour_type="density", variable_data=None,
               remove_outside_points=False):
    """
    Plot a single ternary plot on the given axis.
  
    Parameters:
    ax: The Matplotlib axis object where the plot will be drawn.
    data: The ternary data coordinates to be plotted.
    color (optional, default blue): Color of the data point markers.
    marker (optional, default 'o'): Marker style for data points.
    label (optional, default 'Ternary Plot'): Label for the plot.
    angle (optional, default 0): Rotation angle in degrees for the plot.
    center (optional, default [0, 0]): The center point to rotate around.
    magnifications (optional, default 1): Value of magnification.
    ticks (optional, default [["L", 0], ["R", 0], ["T", 0]]): Ticks for the left, right, and top axes.
    colors (optional, default ["red", "blue", "green"]): Colors for the ternary plot lines.
    linews (optional, default [.8, .8, .8]): Line widths for the ternary plot lines.
    shift_x (optional, default 0): Shift along the x-axis.
    shift_y (optional, default 0): Shift along the y-axis.
    num_lines (optional, default 10): Number of grid lines in the plot.
    labels (optional, default False): Whether to display labels on the axes.
    contours (optional, default False): Whether to display contour plots.
    contour_levels (optional, default 10): Number of contour levels for contouring.
    data_marker_size (optional, default 50): Size of the data point markers.
    cmap (optional, default 'viridis'): Colormap to use for the contour plot.
    coord_system (optional, default 'Ternary'): Coordinate system to use for the plot. Can be 'Ternary' or 'Cartesian'.
    segment_line_color (optional, default 'black'): Color of the segment lines.
    segment_line_width (optional, default 1.0): Width of the segment lines.
    left_label_color (optional, default 'red'): Color for the left axis labels.
    right_label_color (optional, default 'blue'): Color for the right axis labels.
    top_label_color (optional, default 'green'): Color for the top axis labels.
    edge_color (optional, default 'blue'): Color for the edges of the ternary plot.
    """

    while len(linews) < 3:
        linews.append(linews[-1])

    tlines_output = Tlines(ax, num_lines, ticks, colors, linews, angle, center,
                           shift_x, shift_y, magnifications)
   
    edgs(ax, ticks, angle, center, shift_x, shift_y, magnifications)
   
    cart_coords = np.array(TtB(data))
   
    transformed_coords = transform_coordinates(cart_coords, angle, center,
                                            shift_x, shift_y, magnifications)
   
    ternary_vertices = edgs(ax, ticks, angle, center, shift_x, shift_y,
                            magnifications)
   
    
    point_verification = points_in_ternary(ternary_vertices, transformed_coords)
    point_verification = np.array(point_verification)
    
    if remove_outside_points:
        cart_coords = cart_coords[point_verification]
        transformed_coords = transformed_coords[point_verification]
        valid_coords = transformed_coords
    
        if contour_type.lower() == 'variable' and variable_data is not None:
            variable_data = np.array(variable_data)
            if len(variable_data) == len(point_verification):
                valid_variable_data = variable_data[point_verification]
            else:
                print("Warning: Length mismatch between data. Skipping contour data filtering.")
                valid_variable_data = variable_data
        else:
            valid_variable_data = variable_data
    else:
        if not point_verification.all():
            print("Point(s) outside of plot. Please check data.")
        valid_coords = transformed_coords
        valid_variable_data = variable_data
   
    if contours:
        valid_coords = transformed_coords
        point_verification = points_in_ternary(ternary_vertices, transformed_coords)
        point_verification = np.array(point_verification)
    
        x = valid_coords[:, 0]
        y = valid_coords[:, 1]
    
        if contour_type.lower() == 'density':
            x_min, x_max = ternary_vertices[:, 0].min(), ternary_vertices[:, 0].max()
            y_min, y_max = ternary_vertices[:, 1].min(), ternary_vertices[:, 1].max()
    
            high_res = 150
            bins_x = np.linspace(x_min, x_max, high_res)
            bins_y = np.linspace(y_min, y_max, high_res)
    
            density, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y])
    
            smoothed_density = gaussian_filter(density, sigma=8)
            smoothed_density /= smoothed_density.max()
    
            grid_x, grid_y = np.meshgrid(
                0.5 * (x_edges[:-1] + x_edges[1:]),
                0.5 * (y_edges[:-1] + y_edges[1:])
            )
    
            levels = np.linspace(0, 1, contour_levels - 3)
    
            contour = ax.contourf(grid_x, grid_y, smoothed_density.T,
                                  levels=levels, cmap=cmap)
    
            triangle_path = Path(ternary_vertices)
            transformed_triangle_path = TransformedPath(triangle_path, ax.transData)
            contour.set_clip_path(transformed_triangle_path, transform=ax.transData)
    
            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label("Density")
    
        elif contour_type.lower() == 'variable' and valid_variable_data is not None:
            if len(valid_variable_data) != len(valid_coords):
                st.warning("⚠️ Length mismatch between filtered variable data and valid coordinates. Skipping variable contour.")
            else:
                z = np.array(valid_variable_data)
                x = valid_coords[:, 0]
                y = valid_coords[:, 1]
    
                x_min, x_max = ternary_vertices[:, 0].min(), ternary_vertices[:, 0].max()
                y_min, y_max = ternary_vertices[:, 1].min(), ternary_vertices[:, 1].max()
    
                grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
                cs = ax.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap=cmap, alpha=0.5)
    
                triangle_path = Path(ternary_vertices)
                transformed_triangle_path = TransformedPath(triangle_path, ax.transData)
                cs.set_clip_path(transformed_triangle_path, transform=ax.transData)
    
                cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05)
                cbar.set_label("Variable Value")
    
        elif variable_data is None or len(variable_data) != len(transformed_coords):
            st.warning("⚠️ contour_data missing or length mismatch. Skipping contour.")
    
        if show_data_with_contours:
            ax.scatter(x, y, c=[color] * len(x), marker=marker, s=data_marker_size)

    scatter_plot = None
    
    if not contours:    
        scatter_plot = ax.scatter(
            transformed_coords[:, 0], 
            transformed_coords[:, 1],
            c=[color] * len(transformed_coords),
            marker=marker, 
            label=label, 
            s=data_marker_size
        )
       
    if labels:
        
        ltlines, rtlines, ttlines = tlines_output[0]
        LS, RS, TS = tlines_output[1]
           
        LS_cart = [transform_coordinates(TtB([ls]), angle, center,
                                               shift_x, shift_y, magnifications  )[0] for ls in LS]
        RS_cart = [transform_coordinates(TtB([rs]), angle, center,
                                               shift_x, shift_y, magnifications  )[0] for rs in RS]
        TS_cart = [transform_coordinates(TtB([ts]), angle, center,
                                               shift_x, shift_y, magnifications  )[0] for ts in TS]
       
        tick_interval = num_lines
        
        left_labels = list(range(0, 101, tick_interval))
        right_labels = list(reversed(list(range(0, 101, tick_interval))))
        top_labels = list(range(0, 101, tick_interval))
       
        remove_left = int((ticks[0][1])/tick_interval)
        remove_right = int((ticks[1][1])/tick_interval)
        remove_top = int((ticks[2][1])/tick_interval)
       
        if remove_left != 0:
            LS_cart = LS_cart[remove_left:]
            left_labels = left_labels[remove_left:]
            TS_cart = TS_cart[:len(LS_cart)]
            top_labels = top_labels[:len(LS_cart)]
           
        if remove_right != 0:
            RS_cart = RS_cart[:-remove_right]
            right_labels = right_labels[:-remove_right]
            LS_cart = LS_cart[:-remove_right]
            left_labels = left_labels[:-remove_right]
           
        if remove_top != 0:
            TS_cart = TS_cart[remove_top:]
            top_labels = top_labels[remove_top:]
            RS_cart = RS_cart[remove_top:]
            right_labels = right_labels[remove_top:]
       
        for (x, y), label in zip(LS_cart, left_labels):
            ax.text(x, y, label, color=left_label_color, fontsize=12 * magnifications, ha='right', va='top')

        for (x, y), label in zip(RS_cart, right_labels):
            ax.text(x, y, label, color=right_label_color, fontsize=12 * magnifications, ha='left', va='center')

        for (x, y), label in zip(TS_cart, top_labels):
            ax.text(x, y, label, color=top_label_color, fontsize=12 * magnifications, ha='right', va='bottom')
        
    line_starts_transformed = []
    line_ends_transformed = []
    
    if coord_system == 'Ternary':
        if line_starts:
            try:
                line_starts_transformed = transform_coordinates(
                    TtB(line_starts), angle, center, shift_x, shift_y, magnifications
                )
            except Exception as e:
                print("Error converting ternary line_starts:", e)

        if line_ends:
            try:
                line_ends_transformed = transform_coordinates(
                    TtB(line_ends), angle, center, shift_x, shift_y, magnifications
                )
            except Exception as e:
                print("Error converting ternary line_ends:", e)

    elif coord_system == 'Cartesian':
        line_starts_transformed = transform_coordinates(
            line_starts, angle, center, shift_x, shift_y, magnifications
        )
        line_ends_transformed = transform_coordinates(
            line_ends, angle, center, shift_x, shift_y, magnifications
        )

    for start, end in zip(line_starts_transformed, line_ends_transformed):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=segment_line_color, 
                linewidth=segment_line_width)
       
    if cartesian_labels:
        coords = [(xx, yy) for xx, yy, _, _ in cartesian_labels]
        transformed = transform_coordinates(coords, angle, center, shift_x, shift_y, magnifications)
        for (x, y), (_, _, txt, lbl_color) in zip(transformed, cartesian_labels):
            ax.text(
                x, y, txt,
                fontsize=cartesian_label_style.get("fontsize", 12),
                color=lbl_color,
                ha='center', va='center')

    return scatter_plot

#============================================================================

def main():
    
    st.title("Ternary Chart Visualization Tool")
    st.write('''
Use this application to create 1-4 ternary charts.  
:gray[*Please note that this application is a work in progress.*]''')

    
    num_charts = st.slider("Number of Ternary Charts", 1, 4, 1)

    data_choice = st.radio("Select Data Input Method", 
                           ["Upload CSV File", "Enter Data Manually"])

    df = None
    charts_data = {}
    valid_data = True

    if data_choice == "Upload CSV File":
        st.write('''
Upload a .csv file with ternery coords.  
Each column should contain the coords for one ternary chart.  
Each cell should be in the form: x, y, z, and should add to 1.

Example:
    
Ternary_Chart_1

0.33,0.33,0.34

0.81,0.09,0.10''')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.write("File uploaded...")
            df = pd.read_csv(uploaded_file, skipinitialspace=True)
            st.write("Data Preview:", df.head(6))
        else:
            st.write("Waiting for file upload...")
            
    elif data_choice == "Enter Data Manually":
        st.write("Enter ternary coordinates manually in the following format:")
        st.code('data1 = [[0.33, 0.33, 0.34], [0.31, 0.34, 0.35], ...]', language='python')
        
        valid_data = True
        charts_data = {}
        
        for i in range(num_charts):
            st.write(f"### Plot {i + 1} Data")
        
            manual_input = st.text_area(f"Enter Data for Plot {i + 1}", height=150)
            
            if manual_input.strip():
                
                try:
                    start = manual_input.find('[')
                    end = manual_input.rfind(']')
        
                    if start != -1 and end != -1:
                        data_str = manual_input[start:end + 1]
                        chart_data = eval(data_str)
        
                        valid_chart = True
                        for row in chart_data:
                            if len(row) == 3:
                                if abs(sum(row) - 1) >= 1e-5:
                                    st.warning(f"⚠️ Plot {i + 1}: Coordinates {row} do not sum to 1.")
                                    valid_chart = False
                            else:
                                st.error(f"Plot {i + 1}: Invalid row length: {row}")
                                valid_chart = False
        
                        if valid_chart:
                            st.success(f"Plot {i + 1}: Data parsed successfully!")
                            charts_data[f"Plot_{i + 1}"] = chart_data
                        else:
                            valid_data = False
                    else:
                        st.error(f"Plot {i + 1}: Invalid format. Use full Python list syntax.")
        
                except Exception as e:
                    st.error(f"Plot {i + 1}: Error parsing data: {e}")
                    valid_data = False
            
    if df is not None or charts_data:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        chart_settings = []
        colormap_options = plt.colormaps()
        
        with st.sidebar.expander("Chart Options (entire plot)", expanded=True):
            show_title = st.checkbox("Show chart title", value=True, key="show_title_main")
        
            if show_title:
                st.text_input("Chart title text", value="Ternary Plot(s)", 
                              key="chart_title_text")
                st.slider("Title font size", min_value=10, max_value=36, 
                          value=16, key="chart_title_fontsize")
                st.selectbox("Title alignment", ["left", "center", "right"], 
                             index=1, key="chart_title_halign")
                st.slider("Title Y offset", min_value=0.9, max_value=1.2, 
                          value=1.02, step=0.01, key="chart_title_yoffset")
                
        for i in range(num_charts):
            if df is None and (not charts_data.get(f"Plot_{i + 1}") or len(charts_data.get(f"Chart_{i + 1}")) == 0):
                st.warning(f"Plot {i + 1}: No valid data; skipping customization options.")
                continue
            
            with st.sidebar.expander(f'Customization Options - Plot {i+1}', expanded=(i==0)):
                
                # --- Plot Choice --- 
                st.markdown("### Plot Choices")
                
                if df is not None:
                    columns = df.columns.tolist()
                    col = st.selectbox(f"Select column for plot {i+1}", columns, key=f'col_{i}')
                else:
                    col = f"Manual Data {i+1}"
                    
                show_legend = st.checkbox("Show legend", value=True, key=f"show_legend_{i}")
                
                # --- Plot Line Choices ---
                st.markdown("### Plot Lines")
                num_lines = st.selectbox(f"Number of Grid Lines - Plot {i+1}", 
                                     [1, 5, 10], index=2, key=f'num_lines_{i}')
                line_width = st.slider(f"Grid Line Width - Plot {i+1}", 
                                       0.5, 5.0, 0.8, key=f'line_width_{i}')
                line_color_1 = st.color_picker(f"Grid Line Color 1 - Plot {i+1}", 
                                               "#FF0000", key=f'lc1_{i}')
                line_color_2 = st.color_picker(f"Grid Line Color 2 - Plot {i+1}", 
                                               "#0000FF", key=f'lc2_{i}')
                line_color_3 = st.color_picker(f"Grid Line Color 3 - Plot {i+1}", 
                                               "#008000", key=f'lc3_{i}')
                edge_color = st.color_picker(f"Edge Color - Plot {i+1}", 
                                               "#0000FF", key=f'edge_color_{i}')
                
                # --- Transformation Choices ---
                st.markdown("### Transformations")
                shift_x = st.number_input(f"X Shift - Plot {i+1}", 
                                          value=0, step=1, key=f'shift_x_{i}')
                shift_y = st.number_input(f"Y Shift - Plot {i+1}", 
                                          value=0, step=1, key=f'shift_y_{i}')
                angle = st.slider(f"Rotation Angle (degrees) - Plot {i+1}", 
                                  0, 360, 0, key=f'angle_{i}')
                magnification = st.slider(f"Magnification - Plot {i+1}",
                                          min_value=0.25, max_value=2.0,
                                          step=0.05, value=1.0, key=f'magnification_{i}')
                
                # --- Tick Choices ---
                left_tick = st.slider(f"Left Tick - Plot {i+1}", min_value=0, 
                        max_value=100, value=0, step=num_lines, key=f'left_tick_{i}')
                right_tick = st.slider(f"Right Tick - Plot {i+1}", 
                        min_value=0, max_value=100, value=0, step=num_lines, key=f'right_tick_{i}')
                top_tick = st.slider(f"Top Tick - Plot {i+1}", min_value=0, 
                        max_value=100, value=0, step=num_lines, key=f'top_tick_{i}')
                ticks = [["L", left_tick], ["R", right_tick], ["T", top_tick]]
                
                # --- Plot Label Choices ---
                st.markdown("### Plot Labels")
                labels = st.checkbox(f"Show Axis Labels - Plot {i+1}", 
                                     value=False, key=f'labels_{i}')
                left_label_color = st.color_picker(f"Bottom Label Color - Plot {i+1}",
                                        "#0000FF", key=f"left_label_color_{i}")
                right_label_color = st.color_picker(f"Right Label Color - Plot {i+1}", 
                                        "#008000", key=f"right_label_color_{i}")
                top_label_color = st.color_picker(f"Left Label Color - Plot {i+1}", 
                                        "#FF0000", key=f"top_label_color_{i}")
                corner_label_choice = st.checkbox(f"Show Corner (Axis) Labels - Plot {i+1}", 
                                        value=False, key=f'corner_labels_{i}')
                corner_label_colors = {
                    'A': st.color_picker(f"Corner Label A Color - Plot {i+1}", 
                                         "#FF0000", key=f"corner_label_a_color_{i}"),
                    'B': st.color_picker(f"Corner Label B Color - Plot {i+1}", 
                                         "#0000FF", key=f"corner_label_b_color_{i}"),
                    'C': st.color_picker(f"Corner Label C Color - Plot {i+1}", 
                                         "#008000", key=f"corner_label_c_color_{i}"),}
                
                # --- Contour Choices ---
                st.markdown("### Contours")
                contours = st.checkbox(f"Enable Contours - Plot {i+1}", 
                                       value=False, key=f'contours_{i}')
                contour_levels = st.number_input(f"Contour Levels - Plot {i+1}",
                                                 1, 50, 15, key=f'contour_levels_{i}')
                cmap = st.selectbox(f"Colormap - Plot {i+1}", colormap_options, 
                                    index=colormap_options.index("Blues"), key=f'cmap_{i}')
                show_data_with_contours = st.checkbox(f"Show Data Points with Contours - Plot {i+1}", 
                                                      value=True, key=f"show_data_with_contours_{i}")
                contour_type = st.selectbox(f"Contour Type - Plot {i+1}", options=["density", "variable"],
                                            index=0, key=f'contour_type_{i}')
                variable_data_input = ""
                if contour_type == "variable":
                    variable_data_input = st.text_area(
                        f"Variable Values (comma-separated) - Plot {i+1}",
                        help="Enter a list of numeric values corresponding to each data point",
                        key=f"variable_values_{i}")
                
                # --- Marker Choices ---
                st.markdown("### Markers")
                marker_color = st.color_picker(f"Marker Color - Plot {i+1}", 
                                               "#0000FF", key=f'marker_color_{i}')
                marker_style = st.selectbox(f"Marker Style - Plot {i+1}", 
                                ["o", "s", "^", "D", "*"], key=f'marker_style_{i}')
                marker_size = st.slider(f"Marker Size - Plot {i+1}", 
                                        10, 200, 50, key=f'marker_size{i}')
                remove_outside_points = st.checkbox(f"Remove Outside Points - Plot {i+1}", 
                                                value=True, key=f'remove_outside_points_{i}')
                
                # --- Segment Lines ---
                st.markdown("### Segment Lines")
                coord_system = "Cartesian"
                st.write("Segment Lines will be interpreted as Cartesian coordinates (e.g., 50,0 -> 50,86).")
                line_segments_input = st.text_area(f"Line Segments - Plot {i+1} (one per line)",
                    key=f'line_segments_input_{i}')
                segment_line_color = st.color_picker(f"Segment Line Color - Plot {i+1}", 
                                                     "#000000", key=f"segment_line_color_{i}")
                segment_line_width = st.slider(f"Segment Line Width - Plot {i+1}", 
                                               0.5, 5.0, 1.0, key=f"segment_line_width_{i}")
                
                # --- Segment Labels ---
                st.markdown("### Segment Labels")
                num_seg_labels = st.number_input(
                    f"How many segment labels for Plot {i+1}?", 
                    min_value=0, max_value=10, value=0, step=1, key=f"nlabels_{i}")
                
                segment_labels = []

                for j in range(num_seg_labels):
                    c1, c2, c3, c4 = st.columns([2,1,1,1])
                    with c1:
                        text = st.text_input(f"Label text #{j+1}", key=f"text_{i}_{j}")
                    with c2:
                        x = st.number_input(f"X#{j+1}", key=f"x_{i}_{j}", format="%.2f")
                    with c3:
                        y = st.number_input(f"Y#{j+1}", key=f"y_{i}_{j}", format="%.2f")
                    with c4:
                        lbl_color = st.color_picker(f"Color#{j+1}", key=f"color_{i}_{j}")
                    if text:
                        segment_labels.append((x, y, text, lbl_color))
                                
                chart_settings.append({
                        "col": col,
                        "shift_x": shift_x, "shift_y": shift_y, "angle": angle,
                        "magnification": magnification, "ticks": ticks,
                        "colors": [line_color_1, line_color_2, line_color_3],
                        "marker_color": marker_color, "marker_style": marker_style,
                        "marker_size": marker_size, "labels": labels, 
                        "contours": contours, "contour_levels": contour_levels,
                        "show_data_with_contours": show_data_with_contours,
                        "contour_type": contour_type, "variable_data_input": variable_data_input,
                        "num_lines": num_lines, "line_width": line_width, "cmap": cmap,
                        "coord_system": coord_system, "left_label_color": left_label_color, 
                        "right_label_color": right_label_color, "top_label_color": top_label_color,
                        "edge_color": edge_color, "coord_system": coord_system,
                        "line_segments_input": line_segments_input, "segment_line_color": segment_line_color,
                        "segment_line_width": segment_line_width,
                        "cartesian_labels": segment_labels, "corner_label_choice": corner_label_choice,
                        "corner_label_colors": corner_label_colors, 
                        "remove_outside_points": remove_outside_points,})
            
        errors = set()
        
        show_legend_overall = any(st.session_state.get(f"show_legend_{j}", True) for j in range(num_charts))
        
        if 'centers' not in st.session_state:
            st.session_state.centers = {}
        
        for i, settings in enumerate(chart_settings):
            col_data = []

            if df is not None:
                col_data = df[settings["col"]].dropna().astype(str).tolist()
            else:
                col_data = charts_data.get(f"Chart_{i + 1}", [])

            new_col_data = []
            for coord in col_data:
                try:
                    if isinstance(coord, list):
                        parts = coord
                    else:
                        parts = [float(x.strip()) for x in coord.split(',')]
                    if len(parts) == 3:
                        new_col_data.append(parts)
                    else:
                        st.warning(f"Skipping invalid row (incorrect number of coordinates): {coord}")
                except ValueError:
                    st.warning(f"Skipping invalid row (could not convert to float): {coord}")
                                    
            if not new_col_data:
                st.warning(f"Chart {i + 1}: No valid data available, skipping plot.")
                continue
                    
            chart_key = f"chart_{i}_center_{settings['shift_x']}_{settings['shift_y']}"

            if chart_key not in st.session_state.centers:
                center = compute_ternary_center(ax, settings["shift_x"], settings["shift_y"])
                st.session_state.centers[chart_key] = center
            else:
                center = st.session_state.centers[chart_key]
            
            line_starts = []
            line_ends = []
            if settings["line_segments_input"].strip():
                for row in settings["line_segments_input"].strip().splitlines():
                    if '->' in row:
                        start_str, end_str = row.split('->')
                        start_str = start_str.strip()
                        end_str = end_str.strip()
                        try:
                            if settings["coord_system"] == "Cartesian":
                                start_vals = [float(x.strip()) for x in start_str.split(',')]
                                end_vals   = [float(x.strip()) for x in end_str.split(',')]
                                if len(start_vals) == 2 and len(end_vals) == 2:
                                    line_starts.append(start_vals)
                                    line_ends.append(end_vals)
                                else:
                                    st.warning(f"Invalid cartesian line segment: {row}")
                        except ValueError:
                            st.warning(f"Invalid line segment: {row}")
            
            settings["line_starts"] = line_starts
            settings["line_ends"] = line_ends
            
            plot_success = True
            
            for error in errors:
                st.warning(error)
                
            variable_data = []
            if settings["contour_type"] == "variable" and settings["variable_data_input"].strip():
                try:
                    variable_data = [float(v.strip()) for v in settings["variable_data_input"].split(",") if v.strip()]
                except ValueError:
                    st.warning(f"⚠️ Chart {i+1}: Some variable contour values could not be converted to float.")

            if settings["corner_label_choice"]:
                corner_labels = [['A', [-8, -3]], ['B', [108, -3]], ['C', [50, 93]]]
                for label, coord in corner_labels:
                    coords = np.array([[coord[0], coord[1]]])
                    coords = transform_coordinates(
                        coords,
                        angle=settings["angle"],
                        center=center,
                        shift_x=settings["shift_x"],
                        shift_y=settings["shift_y"],
                        magnifications=settings["magnification"],
                    )
                    x, y = coords[0]
                    ax.text(
                        x, y, label,
                        ha='center', va='center',
                        fontsize=12,
                        fontweight='bold',
                        color=settings["corner_label_colors"][label])

            try:
                plot_on_ax(
                    ax, new_col_data, 
                    color=settings["marker_color"],  
                    marker=settings["marker_style"],
                    label=f"Ternary Chart {i+1}",
                    center=center,
                    angle=settings["angle"],
                    magnifications=settings["magnification"],
                    coord_system=settings["coord_system"],
                    ticks=settings["ticks"],
                    shift_x=settings["shift_x"], shift_y=settings["shift_y"], 
                    data_marker_size=settings["marker_size"],
                    labels=settings["labels"], contours=settings["contours"], 
                    contour_levels=settings["contour_levels"],
                    num_lines=settings["num_lines"], colors=settings["colors"], 
                    linews=[settings["line_width"]],
                    cmap=settings["cmap"],
                    line_starts=settings["line_starts"],
                    line_ends=settings["line_ends"],
                    segment_line_color=settings["segment_line_color"],
                    segment_line_width=settings["segment_line_width"],
                    left_label_color=settings["left_label_color"],
                    right_label_color=settings["right_label_color"],
                    top_label_color=settings["top_label_color"],
                    edge_color=settings["edge_color"],
                    cartesian_labels=settings.get("cartesian_labels", []),
                    cartesian_label_style=settings.get("cartesian_label_style", {}),
                    show_data_with_contours=settings["show_data_with_contours"],
                    contour_type=settings["contour_type"], 
                    remove_outside_points=settings["remove_outside_points"],
                    variable_data=variable_data)
                
            except ValueError as e:
                st.warning(f"⚠️ Customization Error in Chart {i+1}: {e} Please change input and try again.")
                plot_success = False
                
        if st.session_state.get("show_title_main", True):
            fig.suptitle(
                st.session_state.get("chart_title_text", "Ternary Plot(s)"),
                fontsize=st.session_state.get("chart_title_fontsize", 16),
                ha=st.session_state.get("chart_title_halign", "center"),
                y=st.session_state.get("chart_title_yoffset", 1.02))

        if plot_success:
            if show_legend_overall:
                ax.legend()
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.axis('off')
            ax.set_axis_off()
            st.pyplot(fig)

            if st.button("Export Image"):
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
                img_bytes.seek(0)
            
                st.download_button(
                    label="Download Image",
                    data=img_bytes,
                    file_name="ternary_charts_export.png",
                    mime="image/png"
                )
                st.success("Plot is ready for download. Select the Download Image button to continue.")
                    
    else:
        st.write("Please upload a CSV file or enter data manually.")

if __name__ == "__main__":
    main()
