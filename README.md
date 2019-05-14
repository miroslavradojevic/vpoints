## Vanishing point detection
#### Implementation using Python and OpenCV library
Script **vpdetector.py** implements the method.  

**vpdetector.py** takes up to 5 parameters (format: --parameterName value) and compulsory image path
If a parameter is not submitted, default value is used. 

###### Usage 
```console
python vpdetector.py -h
Detect vanishing points in jpg image.

positional arguments:
  img                   Image jpg file

optional arguments:
  -h, --help            show this help message and exit
  --cannyMin [CANNYMIN]
                        Canny edge detector: lower threshold [0, inf], default
                        100
  --cannyMax [CANNYMAX]
                        Canny edge detector: upper threshold [0, inf], default
                        300
  --houghTreshold [HOUGHTRESHOLD]
                        Hough t. threshold [0, inf], default 100
  --houghMinLineLen [HOUGHMINLINELEN]
                        Hough t. min. line length [0, inf], default 50
  --scoreThreshold [SCORETHRESHOLD]
                        Score threshold [0, 1], default 0.8

```
Example calls:
```console
python vpdetector.py img "C:\Users\10250153\stack\vpoints\demo\5D4L1L1D_L.jpg" --cannyMin 80 --cannyMax 200 --houghTreshold 50 --houghMinLineLen 30 --scoreThreshold 0.8
python vpdetector.py img "C:\Users\10250153\stack\vpoints\demo\5D4L1L1D_L.jpg" --cannyMin 80 --houghTreshold 50 --houghMinLineLen 30 --scoreThreshold 0.75
```
All default parameter values
```console
python vpdetector.py img "C:\Users\10250153\stack\vpoints\demo\5D4L1L1D_L.jpg"
```
[Demos and data](https://miroslavradojevic.stackstorage.com/s/bMmDUMfDsiHau6o)

[Reseach materials](https://miroslavradojevic.stackstorage.com/s/aGTI3Q4R5NY5SWJ)

[Algorithm overview](https://miroslavradojevic.stackstorage.com/s/JBZf9Z0dy7GYcwH)

[Intersection point weight computation](https://miroslavradojevic.stackstorage.com/s/diCY5KyNmYhyNrB)


