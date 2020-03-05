# Extracting Moving Objects from IR Image Sequences

## Intro
This is an implementation of the 2017 paper [**A Flexible Algorithm for Detecting Challenging Moving Objects in Real-Time within IR Video Sequences**](https://www.mdpi.com/2072-4292/9/11/1128) by Andrea Zingoni, Marco Diani, and Giovanni Corsini of the University of Pisa, and the Italian Naval Academy.

The goal is to implement an algorithm in Python that will distinguish a moving object from the backround, using estimation techniques and user-controlled thresholds.



## TO-DO
- [x] Frame Grabber
  -  Aqcuires frame [n]
- [x] Drift Removal
  -  Corrects the image to compensate for intensity drifts
- [x] Initialize BG Estimation Filter
  -  Average corrected image over [q] frames
- [x] Initialize Variance Estimation Filter
  -  Like BG Initialize but calculating Standard Deviation
- [x] Implement Run-time BG Estimation
- [x] Implement Run-time Variance Estimation
- [x] Add T1 Threshold for Estimation control
- [x] Add T2 Threshold for Output sensitivity
- [ ] Add Refresh of Frozen Estimations
  -  Prevents Permanent 'movement' in a stationary location
- [ ] Output Refinement
  -  Eliminates all single pixel estimations
  -  Dilate -> Erode joins close blobs together

## References

>Zingoni, A.; Diani, M.; Corsini, G. A Flexible Algorithm for Detecting Challenging Moving Objects in Real-Time within IR Video Sequences. Remote Sens. 2017, 9, 1128.
