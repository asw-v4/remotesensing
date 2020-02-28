"""
NOTATION
--------

iBG is Image Background Pixels [x,y,n]
iMV is Moving Target Pixels [x,y,n]
iNO is Image Noise Pixels [x,y,n]
iTOT is the  entire video sequence [x,y,n]
iRES residual image after stripping iBG [x,y,n]

iCOR is the corrected input imaged [x,y,n]
***Can be Negative, vals now have no meaning apart from relative difference***

[x,y] represents a single pixel within frame n

H0hyp is estimated pixels WITH NO movement
H1hyp is estimated pixels WITH movement

SAT is Spatial Average Intensity

alpha is the effective length of the IIR
alpha can adjust on the fly based on the estimated amount of BG movement

T1 is the threshold calc'd frame by frame
sigma[x,y,n] is standard deviation of iCOR[x,y,n]
t1 is tuneable param - LOW = noise HIGH = low contrast iMV


WORKINGS
--------
H0hyp : iTOT[x,y,n] = iBG[x,y,n]+iNO[x,y,n]
H1hyp : iTOT[x,y,n] = iMO[x,y,n]+iNO[x,y,n]

SAT = 1/(NumSeqRow * NumSeqCol)
iCOR[x,y,n] = iTOT[x,y,n] - SAT

# BELOW IS AN EXAMPLE IMPLEMENTATION of calc'd alpha
# alpha = exp(( ln*(1-P)-1 ) / 2Np)
# ##DETERMINE P and Np##

iBG[x,y,n+1] = alpha*iBG[x,y,n] + (1-alpha)*iCOR[x,y,n+1]

T1[x,y,n] = t1*sigma[x,y,n]

T4::

iRES = iCOR[x,y,n] - iBG[x,y,n-1]


WORKFLOW
--------
1. Drift Removal
    -   Calc SAT
    -   Calc iCOR

1.5 Initialize BG Filter
    -   Collect q frames iCOR[x,y,(n0..nq)] frames
    -   average frames to calc iBG[x,y,0]

2. Background Estimation Filter
    -   Calc P ?
    -   Calc alpha
    -   Calc iBG if iCOR[x,y,n+1] - iBG[x,y,n] > T1


"""
