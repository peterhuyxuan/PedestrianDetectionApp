# Pedestrian Detection Application

•	Developed a Computer Vision Python solution to detect and track multiple pedestrians individually and in groups through multiple iterative photo frames  

•	Extracted the features of the pedestrians using Faster Region-Convolutional Neural Network Deep Learning Algorithms (R-CNN) using TensorFlow

•	Used a Centroid based algorithm to find the centre points of the pedestrians in order to keep track and ID the pedestrian in real-time

## Running the operation

Only works up to Python 3.6 due to TensorFlow usage compatability

**Activate the virtual environment**

```
pedestrian-env/Scripts/activate
```

Using activating virtual environment using Git Bash

```
. pedestrian-env/Scripts/activate
```

**Install all packages**

```
pip install -r requirements.txt
```

**Running the program**

Make sure to activate virtual environment.

```
python pedestrian_detection.py
```

The final video output is found in:

```
final_pedestrian_output_vid.avi
```
