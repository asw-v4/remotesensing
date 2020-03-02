# Extracting Moving Objects from Thermal/IR Image Sequences

## Intro
This is an implementation of the 2017 paper [**A Flexible Algorithm for Detecting Challenging Moving Objects in Real-Time within IR Video Sequences**](https://www.mdpi.com/2072-4292/9/11/1128) by Andrea Zingoni, Marco Diani, and Giovanni Corsini of the University of Pisa, and the Italian Naval Academy.

The goal is to implement an algorithm in Python that will distinguish a moving object from the backround, using estimation techniques and user-controlled thresholds.



## TO-DO
- [] Frame Grabber
  -  Aqcuires frame [n]
- [] Drift Removal
  -  Corrects the image to compensate for intensity drifts
- [] Initialize BG Estimation Filter
  -  Average corrected image over [q] frames
- [] Initialize Variance Estimation Filter
  -  Like BG Initialize but calculating Standard Deviation
- [] Implement Run-time BG Estimation
- [] Implement Run-time Variance Estimation
- [] Add T1 Threshold for Estimation control
- [] Add T2 Threshold for Output sensitivity
- [] Add Refresh of Frozen Estimations
  -  Prevents Permanent 'movement' in a stationary location
- [] Output Refinement
  -  Eliminates all single pixel estimations
  -  Dilate -> Erode joins close blobs together

## References

>Zingoni, A.; Diani, M.; Corsini, G. A Flexible Algorithm for Detecting Challenging Moving Objects in Real-Time within IR Video Sequences. Remote Sens. 2017, 9, 1128.
