# %% Q2.2.1
import cv2


# %%
def matchPics(I1, I2):

## Convert images to grayscale, if necessary
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
## Detect features in both images
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp1 = fast.detect(I1,None)
    img1 = cv2.drawKeypoints(I1, kp1, None, color=(255,0,0))

    kp2 = fast.detect(I2,None)
    img2 = cv2.drawKeypoints(I2, kp2, None, color=(255,0,0))


## Obtain descriptors for the computed feature locations
    kp1, des1 = brief.compute(I1, kp1)
    kp2, des2 = brief.compute(I2, kp2)

## Match features using the descriptors
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(I1,kp1,I2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('2.2.1.png', img3)

    return kp1, kp2, matches
