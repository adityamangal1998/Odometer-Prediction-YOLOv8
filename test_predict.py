import os
import cv2
import pandas as pd
import json
from ultralytics import YOLO
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)


def load_model(model_path):
    """Load the YOLO model from the specified path."""
    return YOLO(model_path)


def process_image(image, model, reader, iou_threshold):
    """Process an image to predict the odometer reading."""
    results = model.predict(image, imgsz=640, conf=0.5, iou=iou_threshold)
    output = json.loads(results[0].tojson())
    for out in output:
        if out['name'] == 'odometer':
            coordinates = out['box']
            x1, y1, x2, y2 = int(coordinates['x1']), int(coordinates['y1']), int(coordinates['x2']), int(
                coordinates['y2'])
            output_image = image[y1:y2, x1:x2]
            result = reader.readtext(output_image, detail=0)
            if result:
                return ''.join(result)
    return None


def main():
    """Main inference pipeline."""
    base_path = 'model'  # Update this path as needed
    image_base_path = './data/images'  # Update this path as needed
    model_path = os.path.join(base_path, 'best.pt')
    output_file = 'output.csv'
    iou_threshold = 0.7

    # Load the model
    model = load_model(model_path)

    # Prepare for processing
    output_data = []

    for image_file in os.listdir(image_base_path):
        image_path = os.path.join(image_base_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_file}")
            continue

        prediction = process_image(image, model, reader, iou_threshold)
        output_data.append([image_file, prediction])

    # Save predictions to a CSV file
    df = pd.DataFrame(output_data, columns=['image name', 'prediction'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    main()