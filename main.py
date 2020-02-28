"""
NOTATION
--------
n is frame number
iBG is Image Background Pixels [x,y,n]
iMV is Moving Target Pixels [x,y,n]
iNO is Image Noise Pixels [x,y,n]
iTOT is the  entire video sequence [x,y,n]
[x,y] represents a single pixel within frame n

H0hyp is estimated pixels WITH NO movement
H1hyp is estimated pixels WITH movement

H0hyp : iTOT[x,y,n] = iBG[x,y,n]+iNO[x,y,n]
H1hyp : iTOT[x,y,n] = iMO[x,y,n]+iNO[x,y,n]

SAT (Spatial Average Intensity):
SAT = 1/(NumSeqRow x NumSeqCol)

iCOR is the corrected Image/Seq -> iTOT[x,y,n] - SAT
 ***Can be Negative, vals now have no meaning apart from relative difference***




WORKFLOW
--------
1. Drift Removal
    -   Calc SAT
    -   Calc iCOR
2. Background Estimation



"""
