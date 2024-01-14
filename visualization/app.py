import gradio as gr
import numpy as np
import cv2

def mock_detect_streets(image):
    # Simulate street detection by adding a red overlay
    red_overlay = np.full_like(image, (0, 0, 255), dtype=np.uint8)
    combined_image = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    return combined_image

def mock_detect_cars(image):
    # Simulate car detection by adding green dots
    car_count = 0
    for x in range(50, 450, 100):
        for y in range(50, 450, 100):
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
            car_count += 1
    return image, car_count

def process_image(image):
    streets_image = mock_detect_streets(image)
    cars_image, car_count = mock_detect_cars(np.copy(streets_image))
    return image, cars_image, f"Detected {car_count} cars"

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type='numpy'),
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Detection Info")
    ],
    title="Car Detection in Satellite Images",
    description="Upload a satellite image to detect cars. The app first highlights streets and then detects cars on those streets.",
)

if __name__ == '__main__':
    iface.launch()
