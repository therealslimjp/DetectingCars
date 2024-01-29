import cv2 as cv2
import gradio as gr
import pandas as pd
from modelFunctions import *
import numpy as np
import plotly.graph_objects as go

def identifyCars(image):
    #todo image= process_image(image)
    image= convert_to_grayscale(image)
    num_cars, labeled_image = connected_components(image)
    painted_image = paint_cars(labeled_image)
    centroids = calculate_centroids(num_cars, labeled_image)
    distances = calculate_distances(centroids)
    plot= create_plot(centroids, distances, painted_image)
    return plot


with gr.Blocks() as identifyAllCars:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Image, best with 1000*1000px")
            detect_from_image_button = gr.Button("Detect Cars")
            clear_image_button = gr.Button("Reset Image")
        with gr.Column():
            output_plot = gr.Plot(label="Car Plot")

            clear_output_plot_button = gr.Button("Discard Plot")
            # Markdown text for header and description
            # Create Gradio components
        clear_output_plot_button.click(fn=lambda: None, inputs=None, outputs=output_plot)
        clear_image_button.click(fn=lambda:None, inputs=None, outputs=input_image)


        detect_from_image_button.click(
            fn=identifyCars,
            inputs=[input_image],
            outputs=[output_plot]
        )


def create_plot(centroids, distances, colored_image_for_plotly):
    fig = go.Figure()
    img_width, img_height = colored_image_for_plotly.size
    scale_factor = 1
    adjusted_centroids = [(x, img_height - y) for x, y in centroids]

    for i, (x, y) in enumerate(adjusted_centroids):
        hovertext = f"Car {i}<br>" + "<br>".join(
            [f"Distance to Car {j}: {dist:.2f}" for (k, j), dist in distances.items() if k == i] +
            [f"Distance to Car {k}: {dist:.2f}" for (k, j), dist in distances.items() if j == i]
        )
        color = colored_image_for_plotly.getpixel((int(x), img_height - int(y)))
        color_rgba = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'

        fig.add_trace(go.Scatter(x=[x], y=[y], marker=dict(color=color_rgba), mode='markers', hoverinfo='text',
                                 hovertext=hovertext))

    fig.update_xaxes(showgrid=False, visible=False, range=[0, img_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[0, img_height])
    fig.add_layout_image(
        dict(
            source=colored_image_for_plotly,
            xref="x",
            yref="y",
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            sizing="stretch",
            opacity=1,
            layer="below")
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1        ),
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1
        ) # Set the height of the plot
    )

    return fig

def convert_to_grayscale(image):
        # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # make it to 0 and 255
    gray_image[gray_image == 38] = 255
    print(np.unique(gray_image))

    # Check if the image is already binary
    if not np.array_equal(np.unique(gray_image), [0, 38]):
        print("not binary")
        # Apply a threshold to convert it to binary
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    else:
        binary_image = gray_image
    return binary_image

def dfs(image_arr, labels, x, y, current_label):
    # if is out of bounds
    if x < 0 or y < 0 or x >= image_arr.shape[1] or y >= image_arr.shape[0]:
        return
    # if it is not a car or is already labeled, return
    if labels[y, x] != 0 or image_arr[y, x] != 255:
        return
    # else, label it that this pixel is part of the current label
    labels[y, x] = current_label

    # 4-connectivity: left, right, up, down
    dfs(image_arr, labels, x - 1, y, current_label)
    dfs(image_arr, labels, x + 1, y, current_label)
    dfs(image_arr, labels, x, y - 1, current_label)
    dfs(image_arr, labels, x, y + 1, current_label)

def connected_components(image):
    # convert 1 to 255
    image[image == 1] = 255
    labels = np.zeros_like(image)
    label = 1

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # if it is a car and is not labeled yet, search for adjacent pixels
            if image[y, x] == 255 and labels[y, x] == 0:
                dfs(image, labels, x, y, label)
                label += 1

    return label - 1, labels


def generate_colors_for_cars(n):
    return [tuple(np.random.choice(range(1,256), size=3)) for _ in range(n)]



def paint_cars(labeled_image):
    unique_values = np.unique(labeled_image)
    unique_values_without_black = unique_values[unique_values != 0]
    print(unique_values)
    colors = generate_colors_for_cars(len(unique_values_without_black))
    colors = [(0, 0, 0)] + colors
    color_map = {value: color for value, color in zip(unique_values, colors)}
    print(color_map)
    # Create an RGB image
    rgb_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        rgb_image[labeled_image == value] = color
    return PIL.Image.fromarray(rgb_image)

def calculate_centroids(num_cars, labeled_image):
    centroids = []
    for label in range(1, num_cars + 1):  # Starting from 1 as 0 is usually the background
        # Find the coordinates of pixels that belong to the current label
        y, x = np.where(labeled_image == label)

        # Calculate the centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)

        centroids.append((centroid_x, centroid_y))
    # 'centroids' now contains the centroid coordinates for each car
    return centroids


def calculate_distances(centroids):
    distances = {}

    # Calculate pairwise distances
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            # Coordinates of the two centroids
            x1, y1 = centroids[i]
            x2, y2 = centroids[j]

            # Euclidean distance calculation
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Store the distance
            distances[(i, j)] = distance
    return distances


if __name__ == '__main__':
    num_cars, labeled_image = connected_components(np.load("label.npy"))
    painted_image= paint_cars(labeled_image)
    centroids = calculate_centroids(num_cars, labeled_image)
    distances = calculate_distances(centroids)
    print(centroids)
    print(distances)


