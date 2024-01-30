import requests
from io import BytesIO
from PIL import Image
from pyproj import Proj, transform
import math
import plotly.graph_objects as go
from CoordImageLoader import *


# function to convert lat lon to swiss coord system
def transformCoords(lat, lon):
    # Define the input and output coordinate reference systems
    wgs84 = Proj(init='epsg:4326')  # WGS84, the coordinate system for latitude and longitude
    lv95 = Proj(init='epsg:2056')  # EPSG:2056, Swiss CH1903+/LV9

    x, y = transform(wgs84, lv95, lon, lat)

    return x, y


def download_image(x, y, output_folder):
    try:
        for year in [2020, 2021, 2022, 2023]:
            url = f"https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_{year}_{x}-{y}/swissimage-dop10_{year}_{x}-{y}_0.1_2056.tif"
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                # Extracting the image filename from the URL
                filename = url.split("/")[-1]

                with open(f"{output_folder}/{filename}", 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)

                image_path = f"{output_folder}/{filename}"
                pil_image = Image.open(image_path)

                return pil_image

        return None

    except Exception as e:
        print(f"Error loading image for pixels {x},{y}: {e}")


def extract_region(image, x, y, width, height):
    center_x = (math.floor(x*10) % 10000)
    center_y = (math.floor(y*10) % 10000)

    # Ensure the extraction region doesn't go beyond the boundaries of the image
    image_width, image_height = image.size
    top_left_x = max(0, center_x - width // 2)
    top_left_y = max(0, center_y - height // 2)
    bottom_right_x = min(image_width, center_x + width // 2)
    bottom_right_y = min(image_height, center_y + height // 2)

    print(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    # Crop the region from the image
    region = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    return region


def get_image_for_coords(lat, lon):
    x, y = transformCoords(lat, lon)
    x_url = math.floor(x/1000)
    y_url = math.floor(y/1000)

    image = download_image(x_url, y_url, "./downloaded_images")
    if image is None:
        return None
    return extract_region(image, x, y, 1200, 1200)


def filter_map(lat_input, lon_input):
    fig = go.Figure(go.Scattermapbox(
            lat=[lat_input],
            lon=[lon_input],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=20
            ),
        ))

    value_lat = float(lat_input)
    value_lon = float(lon_input)

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=value_lat,
                lon=value_lon
            ),
            pitch=0,
            zoom=18
        ),
    )

    return fig