import ast
from datetime import datetime
from io import BytesIO
import pytz

import cv2
import numpy as np
from PIL import Image
import requests
from ultralytics import YOLO
from ultralytics.utils import yaml_load


config_path = "./ml/config/yolo.yaml"
model_path = yaml_load(config_path)["model_path"]
class_dict = yaml_load(config_path)["classes"]
colours = [
    [255, 0, 0],  # Blue in BGR
    [0, 255, 0],  # Green in BGR
    [0, 0, 255],  # Red in BGR
    [0, 255, 255] # Yellow in BGR
]
melb_lat, melb_long = -37.840935, 144.946457
model_conf_dict = {
    1: (0.1, 0.1),  # Confidence at day & night
    2: (0.1, 0.25),
    3: (0.1, 0.15),
    4: (0.1, 0.1),
    5: (0.1, 0.15),
    6: (0.1, 0.1),
    7: (0.1, 0.1),
    8: (0.1, 0.1),
    9: (0.1, 0.1),
    10: (0.1, 0.15),
    11: (0.1, 0.15),
    12: (0.1, 0.15),
    13: (0.1, 0.1),
    14: (0.1, 0.1)
}


def snap_cam(cam_id):
    """Takes a snapshot of a real-time traffic video at the given camera ID.
    Source: Linkt: https://www.linkt.com.au/using-toll-roads/traffic-and-roadworks/melbourne#live

    Parameters
    ----------
    cam_id: int
        ID of 
    Returns
    -------
    img: obj
        Pillow image object
    """
    cam_id_str = str(cam_id).zfill(2)
    url = f"https://cmlwebcam.transurban.com/wimages/webcam{cam_id_str}.jpg"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content))

    return img


def check_if_daylight(lat, long):
    """Checks if the given coordinate location is at daylight time.
    Source: Sunrise and sunset times API: https://sunrise-sunset.org/api

    Parameters
    ----------
    lat: float
        Latitude
    long: float
        Longitude
    
    Returns
    -------
    boolean
        Indication on whether the location at daylight
    """
    # Fetch Melbourne sunrise & sunset times
    url = f"https://api.sunrise-sunset.org/json?lat={lat}&lng={long}"
    response = requests.get(url)
    results = response.text
    results_dict = ast.literal_eval(results)
    sunrise_time_utc = results_dict["results"]["sunrise"]
    sunset_time_utc = results_dict["results"]["sunset"]

    # Convert to Melbourne timezone
    tz = pytz.timezone('Australia/Melbourne')
    curr_datetime_melb = datetime.now(tz)
    curr_date_str_melb = curr_datetime_melb.strftime("%Y-%m-%d")
    sunrise_datetime_utc = datetime.strptime(
        f"{curr_date_str_melb} {sunrise_time_utc}",
        "%Y-%m-%d %I:%M:%S %p"
    )
    sunrise_datetime_melb = sunrise_datetime_utc.replace(tzinfo=pytz.utc).astimezone(tz)
    sunset_datetime_utc = datetime.strptime(
        f"{curr_date_str_melb} {sunset_time_utc}",
        "%Y-%m-%d %I:%M:%S %p"
    )
    sunset_datetime_melb = sunset_datetime_utc.replace(tzinfo=pytz.utc).astimezone(tz)

    # Check if current Melbourne time is at daylight
    if (curr_datetime_melb >= sunrise_datetime_melb) \
        & (curr_datetime_melb < sunset_datetime_melb):
        return True
    else:
        return False


def load_model(model_path):
    """Loads pre-trained YOLO model with given model path.

    Parameters
    ----------
    model_path: str
        Path to pre-trained model file

    Returns
    -------
    model: class
        YOLO model class object
    """
    model = YOLO(model_path)

    return model


def draw_box(img, class_id, x, y, w, h):
    """Draws a box around the detected vehicle with vehicle type.
    
    Parameters
    ----------
    img:

    class_id: int
        Object class ID
    x: int
        X coordinate of object box
    y: int
        Y coordinate of object box
    w: int
        Width of object box
    h: int
        Height of object box

    Returns
    -------
    None
    """
    label = class_dict[class_id]
    index = list(class_dict).index(class_id)
    colour = colours[index]
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(img, label, (x - 1, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)


def main():
    # Load pre-trained YOLO model
    model = load_model(model_path)

    # Test to adjust model confidence for each traffic camera
    for i in range(14):
        # Convert PIL image to CV image
        pil_img = snap_cam(i+1)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Detect vehicles
        """Due to degraded performance at night, the model confidence
        threshold is to set differently at daylight and night.
        """
        is_daylight = check_if_daylight(melb_lat, melb_long)
        if is_daylight:
            model_conf = model_conf_dict[i+1][0]
        else:
            model_conf = model_conf_dict[i+1][1]
        results = model(
            source=cv_img,
            classes=list(class_dict.keys()),
            conf=model_conf
        )

        # Iterate through output to label and count detected vehicles
        detected_vehicle_dict = dict(zip(class_dict.values(), [0]*len(class_dict)))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Label detected vehicle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                class_id = int(box.cls[0])
                class_name = class_dict[class_id]
                draw_box(cv_img, class_id, x1, y1, w, h)

                # Count detected vehicle type
                detected_vehicle_dict[class_name] += 1

        # Display labelled image with detected vehicles
        print("#############################")
        print("CAM", i+1, "Confidence:", model_conf)
        print(detected_vehicle_dict)
        cv2.imshow("Detected Image", cv_img)
        cv2.waitKey()


if __name__ == "__main__":
    main()
            

