# Detection

## Requirements

- Linux or macOS with Python ≥ 3.7 
- PyTorch ≥ 1.8
- OpenCV

## Setup

1. Clone this repository:
    ```
    https://github.com/elianaddo/detection.git
    ```

## Project Structure 

Within the `projects` folder, you have three pre-trained models ready for use in person detection: YOLOv8, MobileNet, and Detectron2.

## Usage

```
usage: main.py [-h] [-i INPUT] [--ssd] [--det2] [--yolo] [--c1x C1X]
               [--c1y C1Y] [--c2x C2X] [--c2y C2Y] [--confidence CONFIDENCE]
               [--norma NORMA] [--Rx RX] [--Ry RY]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input video file (default: None)
  --mbl
  --det2
  --yolo

  --c1x C1X             X-coordinate of the first point (percentage) (default: 0.0)
  --c1y C1Y             Y-coordinate of the first point (percentage) (default: 60.0)
  --c2x C2X             X-coordinate of the second point (percentage) (default: 100.0)
  --c2y C2Y             Y-coordinate of the second point (percentage) (default: 60.0)

  --confidence          Confidence of the model (default: 0.4)

  --norma NORMA         Maximum distance between centroids

  --Rx RX               X-coordinate of the entry point (percentage) (default: 0.0)
  --Ry RY               Y-coordinate of the entry point (percentage) (default: 0.0)
```
- `-i` The input video file
  
Models:

- `--mbl` Run with mobileNet model
- `--det2` Run with Detectron2 model
- `--yolo` Run with Yolo v8 model

Limit line: 

The "limit line" in the code is used to determine if a person has entered or exited a specific area in a video. When a person crosses this line, the code checks the direction of movement to determine whether they are entering or leaving the monitored area. This line can be placed at any position determined by the user.        
- `--c1x` X-coordinate of the first point of the 'limit line' 
- `--c1y` Y-coordinate of the first point of the 'limit line' 
- `--c2x` X-coordinate of the second point of the 'limit line' 
- `--c2y` Y-coordinate of the second point of the 'limit line'

Entry point: 

The "entry point" specifies where on the limit line represents the entry into the monitored area. It helps determine if a person is entering or exiting based on their position. 
- `--Rx` Set the x-coordinate of the entry point, that is used to determine where the entrance is located (para saber se a pessoa está a entrar ou a sair). 
- `--Ry` Set the y-coordinate of the entry point, that is used to determine where the entrance is located (para saber se a pessoa está a entrar ou a sair). 

Confidence and norma:   
- `--confidence` Sets the confidence threshold for object detection. Only detections with confidence above this value will be considered valid.
- `--norma` Sets a maximum distance between centroids to determinate whether a new detected object should be associated with an existing object based on its position.

## Running

1. Place the video you want to test in the `projects` folder and run the program with the desired flags. 

   ```
    python main.py --input test_1.mp4 --mbl --confidence 0.4 --norma 12 --c1x 0 --c1y 60 --c2x 200 --c2y 0 --Rx 600 --Ry 600 
   ```  

## Integrating Another Model

If you wish to integrate another model into the project, follow these steps:

1. Clone this repository.
2. Add the project of the new model to the `projects` folder.
3. Create an 'api.py' for your new model. That 'api.py' must be able to run a video frame by frame and it needs to return the following values:
   - Confidence
   - Bounding box
   - Class ids
     
     
    ```python
    def drawboundingboxes(frame, totalFrames):   
        # your code goes here 
        return ids, confid, boxes
    ``` 
    
5. Create a function in the main class for your new model with the following structure:

    ``` python
    def _execNewModel(videopath, c1, c2, r, confidence, norma):
    CFG["confidence"] = confidence
    execute_detection(videopath, det2_dboxes, c1, c2, r, norma)      
    ``` 
6. Add a new flag in the parse function to invoke your model.
   ```python
   parser.add_argument("--newModel", dest="newModel", action="store_true")
   ```

7. Update the 'main' function to handle this new flag.
    ```python
    if args.newModel:
        _execNewModel(real_path, c1, c2, r, confidence, norma)
    ```

8. Place the video you want to test in the `projects` folder and run the program with the flags you want:
    ```
    python main.py --input test_1.mp4 --newModel --confidence 0.4 
    ```  
