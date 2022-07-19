import json
import os
import shutil
import math
import numpy as np
import enum
import random


## !! THE GLOBALS BELOW ARE FOR USE SPECIFICALLY WITH ROTATION DATASET CODE
## !! THE GLOBALS FOR WIREFRAME CREATION ARE DIFFERENT!!

# bone is (boneId, parentBoneId, headKeypointId. tailKeypointId)
Bones = [(0,-1,0,1),    (1,0,1,2),     (2,1,2,3),     (3,2,3,4),      \
        (4,-1,0,5),    (5,4,5,6),     (6,5,6,7),     (7,6,7,8),      \
        (8,-1,0,9),    (9,8,9,10),    (10,9,10,11),  (11,10,11,12),  \
        (12,-1,0,13),  (13,12,13,14), (14,13,14,15), (15,14,15,16),  \
        (16,-1,0,17),  (17,16,17,18), (18,17,18,19), (19,18,19,20)]

palmBones    =  [(0,-1,0,1),     (4,-1,0,5),     (8,-1,0,9),  (12,-1,0,13),  (16,-1,0,17)]
thumbBones   =  [(1,0,1,2),      (2,1,2,3),      (3,2,3,4)]
indexBones   =  [(5,4,5,6),      (6,5,6,7),      (7,6,7,8)]
middleBones  =  [(9,8,9,10),     (10,9,10,11),   (11,10,11,12)]
ringBones    =  [(13,12,13,14),  (14,13,14,15),  (15,14,15,16)]
pinkyBones   =  [(17,16,17,18),  (18,17,18,19),  (19,18,19,20)]
 
# 1st val in above is bone id. 2nd val is bone id of parent bone (bone that the bone rotates relative to
# in blender, -1 means no parent). 3rd val is head joint id and 4th is tail

allBones = [palmBones, thumbBones, indexBones, middleBones, ringBones, pinkyBones]
fingers = ["palm", "thumb", "index", "middle", "ring", "pinky"]
colours = ["black", "blue", "green", "magenta", "orange", "red"]

FINGERS                                           = [0, 1, 2, 3, 4, 5]
PALM, THUMB, INDEX, MIDDLE, RING, PINKY           = 0, 1, 2, 3, 4, 5
HEAD, TAIL, X, Y, Z                               = 0, 1, 0, 1, 2
BONE1, BONE2, BONE3, BONE4, BONE5, WRIST          = 0, 1, 2, 3, 4, 0
thumbTip, indexTip, middleTip, ringTip, pinkyTip  = 4, 8, 12, 16, 20  

def X(pose, keyPoint): return pose[0+3*keyPoint]
def Y(pose, keyPoint): return pose[1+3*keyPoint]
def Z(pose, keyPoint): return pose[2+3*keyPoint]

def XYZ(pose, keyPoint): return X(pose, keyPoint), Y(pose, keyPoint), Z(pose, keyPoint)

def createAlphaFolders(folderName, alpha):    
   
    # Create the input foldername and all alpha sub-folders if they don't exist.                      
   
    if os.path.isdir(folderName) is False: os.makedirs(folderName)                
   
    for dir in alpha:    
        if os.path.isdir(folderName + '/' + dir) is False:
            os.makedirs(folderName + '/' + dir)  

def loadPosesFromFile(filename):
   
    #
    # The input file must be json file with "pose" and "label" categories.
    # This function changes the values in the file to convert from the Blender coord system to normal image.
    # The function returns an np array of the poses coords and the corresponding labels.
    #
     
    # Opening JSON file
    file = open(filename,)      
   
    # Returns JSON object as a dictionary
    data = json.load(file)
 
    # Convert the pose data into a np array. keypoints array will be size n x 63 - for n data samples (63 is 21 joints x 3 coords)
    poses = np.array(data["pose"])

    # Extact the labels.
    labels = np.array(data["label"])

    # Extact the metadata.
    if "meta" in data:
        meta = np.array(data["meta"])
    else:
        meta = None    

    print(poses.shape, labels.shape)
   
    # Closing file
    file.close()
   
    return poses, labels, meta  

def convertPosesToOpencv2(poses):

    ##  !!! NEED TO CHECK THIS !!

    # Change the sign of the Y coord values. The blender Y axis is opposite
    # direction to OpenCV / image pixel axes. (Use np slicing. [;, iterates through
    # all rows, 0::3] takes every 3rd value from first col for Xs, then 1::3 for Ys)
   
    poses[:,1::3] = - poses[:,1::3]

    # Swap around the X and Y coords as image coords are +X to the right and +Y to the floor.
    posesX = np.copy(poses[:,0::3])     # save the X coords
    poses[:,0::3] = poses[:,1::3]       # put Ys into Xs
    poses[:,1::3] = posesX              # put saved Xs into Ys

    return poses

def transformPoses_LeftToRight(poses, COPY=True):
    ## CONVERT POSE FROM BLENDER XYZ TO HDR. H IS HIGH, D IS DEEP AND R IS RIGHT
    _poses = np.copy(poses) if COPY else poses
    _poses[:,0::3] = -_poses[:,0::3] # THIS SEEMS TO CONVERT BLENDER COORDS TO VH, HR, D  
    #for _pose in _poses:
    #    _pose[0::3] = -_pose[0::3]  # THIS SEEMS TO CONVERT BLENDER COORDS TO VH, HR, D    
    return _poses
   
def transformPoses_ToUnitPoses(poses, COPY=True):
    ## PUTS POSES IN UNIT CUBES
    _poses = np.copy(poses) if COPY else poses
    _poses = np.copy(poses)
    for _pose in _poses:
        _pose = moveWristToOrigin(_pose)
        _pose = putPoseInUnitCube(_pose)
    return _poses

def transformPoses_AlignPalms(poses, COPY=True):
    ## STRAIGHTENS PALM UPRIGHT AND FLAT ACROSS THE BACK
    _poses = np.copy(poses) if COPY else poses
    _poses = np.copy(poses)
    for _pose in _poses:
        _pose = alignPalm(_pose, False)
    return _poses

def transformPoses_AlignPalms_CV2(poses, COPY=True):
    ## STRAIGHTENS PALM UPRIGHT AND FLAT ACROSS THE BACK
    _poses = np.copy(poses) if COPY else poses
    _poses = np.copy(poses)
    palmAlignments = []
    for _pose in _poses:
        _pose, palmAlignment = alignPalm_CV2cooords(_pose, False)
        palmAlignments.append(palmAlignment)
    return _poses, palmAlignments

def moveWristToOrigin(pose):
    wrist = pose[0:3]
    pose[0::3] = pose[0::3] - wrist[0]
    pose[1::3] = pose[1::3] - wrist[1]
    pose[2::3] = pose[2::3] - wrist[2]
   
    return pose

def pp(flt): # pretty print for floats
   
    return( (int(flt*100)/100) )

def putPoseInUnitCube(pose):
   
    # Find the min and max X and Y coord of all the poses. This is the bounding box within which all pose hands will fit.
    # (Use np slicing. [;, iterates through all rows, 0::3] takes every 3rd value from first col for Xs, then 1::3 for Ys)
    minX = np.min(pose[0::3])
    minY = np.min(pose[1::3])
    minZ = np.min(pose[2::3])
    maxX = np.max(pose[0::3])
    maxY = np.max(pose[1::3])
    maxZ = np.max(pose[2::3])
   
    # The bounding box is the maxmin differences. This is in Blender coord units (metres).
    boundingBoxW = abs(maxX - minX) # width range  !!!  X IS ACTUALLY THE HEIGHT IN BLENDER !!!!
    boundingBoxH = abs(maxY - minY) # height range
    boundingBoxD = abs(maxZ - minZ) # depth range
   
    # the bounding cube side is the largest of the three (hte cube is equal-sided)
    boundingCubeSide = np.max([boundingBoxW, boundingBoxH, boundingBoxD])                        
    #print(boundingBoxH, boundingBoxW, boundingBoxD, boundingCubeSide)
                           
    # Move the pose so that all the points are positive - so in +ive space
    pose[0::3] = pose[0::3] - minX
    pose[1::3] = pose[1::3] - minY
    pose[2::3] = pose[2::3] - minZ
                           
    # Scale the cube to be the unit size eg 1x1x1
    cube = (1,1,1)
    pose[0::3] = (pose[0::3] / boundingCubeSide) * cube[0]     # Scale X coords
    pose[1::3] = (pose[1::3] / boundingCubeSide) * cube[1]     # Scale Y coords
    pose[2::3] = (pose[2::3] / boundingCubeSide) * cube[2]     # Scale z coords    
   
    return pose

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle in radians around a given origin.
    """
    ox, oy = origin
    px, py = point # assumes it is in the xy plane

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
   
    return qx, qy

def rotatePose(origin, pose, angles):
     
    # angles is the euler xyz rotation angles in degrees
   
    rotation_X ,rotation_Y, rotation_Z = angles
    origin_X , origin_Y, origin_Z = origin
       
    # for each point in the pose, rotate around X axis    
    for keypoint in range(0,21):
       
        # get the x,y,z, coords of the point
        X,Y,Z = getPoint(pose, keypoint)
       
        # rotate the point around the X axis (so use the y,z coords of the point and of the origin)
        ZY_rotated = rotate((origin_Z, origin_Y), (Z,Y), math.radians(rotation_X))        
       
        # update the pose with the rotated values before the next axis rotation
        pose[[2+3*keypoint,1+3*keypoint]] = ZY_rotated  

    # for each point in the pose, rotate around Y axis    
    for keypoint in range(0,21):
       
        # get the x,y,z, coords of the point
        X,Y,Z = getPoint(pose, keypoint)
       
        # rotate the point around the Y axis (so use the x,z coords of the point and of the origin)
        XZ_rotated = rotate((origin_X, origin_Z), (X,Z), math.radians(rotation_Y))      
       
        # update the pose with the rotated values before the next axis rotation
        pose[[0+3*keypoint,2+3*keypoint]] = XZ_rotated  

    # for each point in the pose, rotate around Z axis    
    for keypoint in range(0,21):
       
        # get the x,y,z, coords of the point
        X,Y,Z = getPoint(pose, keypoint)
       
        # rotate the point around the Z axis (so use the x,y coords of the point and of the origin)
        YX_rotated = rotate((origin_Y, origin_X), (Y,X), math.radians(rotation_Z))        
       
        # update the pose with the rotated values before the next axis rotation
        pose[[1+3*keypoint,0+3*keypoint]] = YX_rotated  

    return pose

def rotatePoses(poses, rotate_by=0):
   
    # for each pose in the poses
    for poseIndex, pose in enumerate(poses):
                       
        ## Rotate the pose points about the wrist
        wrist_X = int(pose[0])    
        wrist_Y = int(pose[1])
        for keypoint in range(0,21):
            X = pose[3*keypoint]
            Y = pose[1+3*keypoint]                                    
            XY_rotated = rotate( (wrist_X, wrist_Y), (X,Y), math.radians(rotate_by))
            #print("  X, Y, wrist_X, wrist_Y, -rotate_by, XY_rotated ", X, Y, wrist_X, wrist_Y, rotate_by, XY_rotated)
            pose[3*keypoint:2+3*keypoint] = XY_rotated  # :2 doesnt incl 2

    return poses

def getDistance(head, tail):
   
    # this returns the lenght of a vector in 3d from point head to point tail.
    # head and tail are both 3d x,y,z, coords.
   
    X1, Y1, Z1 = head
    X2, Y2, Z2 = tail
   
    diffX = X2 - X1
    diffY = Y2 - Y1    
    diffZ = Z2 - Z1

    distance = math.sqrt( (diffX ** 2) + (diffY ** 2) + (diffZ ** 2) )

    return distance

def getDistancePoints(pose, keypointId1, keypointId2):
   
    # this return the distance between two points of the 21 points of a pose.
    # keypointids passed in to this function are between 0 and 20.

    head = getPoint(pose, keypointId1)
    tail = getPoint(pose, keypointId2)
    distance = getDistance(head, tail)

    return distance

def getDirection(head, tail):
   
    # this returns the euler x,y,z rotations of a vector in 3d from point head to point tail
    # head and tail are both 3d x,y,z, coords
   
    # this return directions in degrees
   
    headX, headY, headZ = head
    tailX, tailY, tailZ = tail
   
    diffX = tailX - headX
    diffY = tailY - headY    
    diffZ = tailZ - headZ
   
    quadX, quadY, quadZ = 0, 0, 0
   
    if diffY > 0 and diffZ >= 0 : quadX = 1
    if diffY > 0 and diffZ <= 0 : quadX = 2  
    if diffY < 0 and diffZ <= 0 : quadX = 3
    if diffY < 0 and diffZ >= 0 : quadX = 4
 
    if diffZ > 0 and diffX >= 0 : quadY = 1
    if diffZ > 0 and diffX <= 0 : quadY = 2  
    if diffZ < 0 and diffX <= 0 : quadY = 3
    if diffZ < 0 and diffX >= 0 : quadY = 4
   
    if diffX > 0 and diffY >= 0 : quadZ = 1
    if diffX > 0 and diffY <= 0 : quadZ = 2  
    if diffX < 0 and diffY <= 0 : quadZ = 3
    if diffX < 0 and diffY >= 0 : quadZ = 4
       
    # get X direction
    if diffY == 0:
        direction_X = 0
    elif diffZ == 0:
        direction_X = 90
    else:
        direction_X = math.degrees(math.atan(diffY/diffZ))
   
    if quadX == 1: direction_X = abs(direction_X)
    if quadX == 2: direction_X = 180 - abs(direction_X)
    if quadX == 3: direction_X = 180 + abs(direction_X)
    if quadX == 4: direction_X = -abs(direction_X)

    # get Y direction
    if diffZ == 0:
        direction_Y = 0
    elif diffX == 0:
        direction_Y = 90
    else:
        direction_Y = math.degrees(math.atan(diffZ/ diffX))
   
    if quadY == 1: direction_Y = abs(direction_Y)
    if quadY == 2: direction_Y = 180 - abs(direction_Y)
    if quadY == 3: direction_Y = 180 + abs(direction_Y)
    if quadY == 4: direction_Y = -abs(direction_Y)
       
    # get Z direction
    if diffX == 0:
        direction_Z = 0
    elif diffY == 0:
        direction_Z = 90
    else:
        direction_Z = math.degrees(math.atan(diffX/diffY))
 
    if quadZ == 1: direction_Z = abs(direction_Z)
    if quadZ == 2: direction_Z = 180 - abs(direction_Z)
    if quadZ == 3: direction_Z = 180 + abs(direction_Z)
    if quadZ == 4: direction_Z = -abs(direction_Z)
   
    #print(quadX, quadY, quadZ)
   
    return direction_X, direction_Y, direction_Z
   
def getPoint(pose, keypointId):
   
    # keyPoint is 0 to 20 keypointId, and KeyPoint_XYZ is the x,y,z coords.
   
    KeyPoint_XYZ = pose[3*keypointId:3+3*keypointId]

    return KeyPoint_XYZ
 
def alignPalm(pose, COPY=True):
   
    _pose = np.copy(pose) if COPY else pose
   
    #
    # THIS TAKES TWO VECTORS, VECTOR1 AND VECTOR2.
    # VECTOR1 IS WRIST TO MIDDLE KNUCKLE. VECTOR2 IS INDEX KNUCKLE TO PINKY KNUCKLE.
    # USE VECTOR1 DIRECTION TO ROTATE THE POSE TO STRAIGHTEN IT.
    # USE VECTOR2 DIRECTION TO MAKE IT FLAT ACROSS THE BACK OF THE PALM.
    #
    # TO STRAIGHTEN VECTOR1, MAKE ITS DIRECTION_Z 90 DEGREES AND DIRECTION_Y 0 DEGREES.
    # TO FLATTEN VECTOR2, MAKE ITS DIRECTION_X -90 DEGREES.
    #
   
    ## 1) rotate the pose so that vector_1 has a z rotation of 90
    v1_direction = getDirection(getPoint(_pose, 0), getPoint(_pose, 9))
    about = getPoint(_pose, 0)
    _pose = rotatePose(about, _pose, (0, 0, 90-v1_direction[2]))

    ## 2) rotate the pose so that vector_1 has a y rotation of 0
    v1_direction = getDirection(getPoint(_pose, 0), getPoint(_pose, 9)) # need to recalc
    about = getPoint(_pose, 0)
    _pose = rotatePose(about, _pose, (0, -v1_direction[1], 0))

    ## 3) rotate the pose so that vector_2 has a x rotation of -90    
    v2_direction = getDirection(getPoint(_pose, 5), getPoint(_pose, 17))
    about = getPoint(_pose, 0)
    _pose = rotatePose(about, _pose, ((-v2_direction[0])-90, 0, 0))
   
    return _pose
   
def alignPalm_CV2cooords(pose, COPY=True):
   
    _pose = np.copy(pose) if COPY else pose
   
    #
    # THIS TAKES TWO VECTORS, VECTOR1 AND VECTOR2.
    # VECTOR1 IS WRIST TO MIDDLE KNUCKLE. VECTOR2 IS INDEX KNUCKLE TO PINKY KNUCKLE.
    # USE VECTOR1 DIRECTION TO ROTATE THE POSE TO STRAIGHTEN IT.
    # USE VECTOR2 DIRECTION TO MAKE IT FLAT ACROSS THE BACK OF THE PALM.
    #
    # TO STRAIGHTEN VECTOR1, MAKE ITS DIRECTION_Z 180 DEGREES AND DIRECTION_X -90 DEGREES.
    # TO FLATTEN VECTOR2, MAKE ITS DIRECTION_Y 180 DEGREES.
    #

    ## 1) rotate the pose so that vector_1 has a z rotation of 180  
    v1_direction = getDirection(getPoint(_pose, 0), getPoint(_pose, 9))
    about = getPoint(_pose, 0)
    _pose = rotatePose(about, _pose, (0, 0, (-v1_direction[2])+180))
    rotate_V1_Z = (v1_direction[2])

    ## 2) rotate the pose so that vector_1 has a x rotation of -90  
    v1_direction = getDirection(getPoint(_pose, 0), getPoint(_pose, 9)) # need to recalc
    about = getPoint(_pose, 0)
    #_pose = rotatePose(about, _pose, ((-v1_direction[0])-90, 0, 0))
    rotate_V1_X = (v1_direction[0])
   
    ## 3) rotate the pose so that vector_2 has a y rotation of 180  
    v2_direction = getDirection(getPoint(_pose, 5), getPoint(_pose, 17))
    about = getPoint(_pose, 0)
    #_pose = rotatePose(about, _pose, (0, (-v2_direction[1])+180, 0))
    rotate_V2_Y = (v2_direction[1])

    ### !!!  17/02/22 CHANGED TO JUST ROTATE UP - DONT STRAIGHTEN OR FLATTEN    


    # return the rotations as well as the pose
    palmAlignments = [rotate_V1_Z, rotate_V1_X, rotate_V2_Y]

    return _pose, palmAlignments

def getPalmRotations(pose):  

    # from mediapipe 20221 paper, 2 vectors are used to give a measure of the palm rotation
    # v1 = middle knuckle to wrist = keypoint 9 to 0
   
    ##
    ##  !!! I USE SLIGHTLY DIFFERENT V1 - FROM WRIST TO MIDDLE KNUCKLE - NOT THE OTHER DIRECTION
    ##

    # v2 = index knuckle to pinky knuckle = keypoint 5 to 17
   
    # so get the direction (euler angles) of v1 and v2
    # and first rotate the pose wrt the negative of v1 - to straighten the palm upright
    # and then rotate the pose wrt the negative of v2 - to twist it around to face the front
   
    # this returns angles in degrees
   
    v1_direction = getDirection(getPoint(pose, 0), getPoint(pose, 9))
    v2_direction = getDirection(getPoint(pose, 5), getPoint(pose, 17))  

    return v1_direction, v2_direction  
   
def resetPlots(plt):
    plt.switch_backend('nbAgg')  # switch back to default in notebook plots
    return

def setPlots(plt):
    plt.switch_backend('TkAgg')  # set to separate window for plots because of plot sequences issue with inline
    return  

def convertPosesToRotations(poses, labels):
 
    ## THIS PUTS A CHAR AT THE END = THE LETTER LABEL . IT SEEMS TO CONVERT ALL NUMERIC VALUES TO STRINGS !!!
    ## I USE createDataset_PoseRotations() NOW INSTEAD
   
    # pose bone directions are their angles of rotation around the axes - can be considered from
    # projections on 2d planes.
    # bone rotations are relative rotations between a pair of bones - namely a bone and its parent.    
   
    posesRotations = []
   
    # for each hand pose in poses
    for poseIndex, pose in enumerate(poses):
       
        label = labels[poseIndex]
           
        poseRotations = []
       
        # first add in the palm alignment vectors, Vector_1, Vector_2 (from MediaPipe paper)    
        for palmRotation in getPalmRotations(pose):
            # append the rotation to the list
            for axis in range(3):
                poseRotations.append(palmRotation[axis])  
                   
        # for each finger in hand
        for finger in FINGERS:
           
            # for each bone in finger
            for bone in allBones[finger]:

                boneId, parentBone, head, tail = bone     # e.g. for index, bone2, we get (6,5,6,7)
                headXYZ = getPoint(pose, head)  
                tailXYZ = getPoint(pose, tail)
               
                if parentBone == -1:  # it's a palm bone                                
                    boneDirection = getDirection(headXYZ, tailXYZ)                  
                    boneRotation = boneDirection
                   
                else:                
                    parentBone, null, parentBonehead, parentBonetail = Bones[parentBone]                          
                    parentHeadXYZ = getPoint(pose, parentBonehead)
                    parentTailXYZ = getPoint(pose, parentBonetail)                                              
                    # get bone euler x,y,z plane rotations - these are wrt the local coord system  
                    boneDirection = getDirection(headXYZ, tailXYZ)

                    # get parent bone euler x,y,z plane rotations - these are wrt the local coord system  
                    parentBoneDirection = getDirection(parentHeadXYZ, parentTailXYZ)
                    # get the bone euler rotation RELATIVE to the parent - IS INDEPENDENT OF LOCAL COORD SYSTEM !  
                    boneRotation = tuple([boneDirection[i] - parentBoneDirection[i] for i in range(3)])

                # append the rotation to the list
                for axis in range(3):
                    poseRotations.append(boneRotation[axis])                    
                   
                #print("bone rotations: ", tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] ) )
 
        # append the label to the end of the list
        # convert the letter to an int = its position in the alphabet

        poseRotations.append(label[0])  
       
        # add the pose rotation into the overall list
        posesRotations.append(poseRotations)
       
    # return an np array      
    return np.array(posesRotations)
           
def deleteLabelPoses(poses, labels, labels2delete):
   
    # this deletes rows in poses for the label letter
   
    _poses = np.copy(poses)
    _labels = np.copy(labels)
    labelLetters = np.array([_labels[i][0] for i in range(_labels.shape[0])])
   
    # delete all poses samples rows for the letters

    #index = np.where(labelLetters==label2delete)[0] # this is the index of the rows to be deleted
   
    index = []
    for labelTarget in labels2delete:
        indexTarget = np.where(labelLetters==labelTarget)[0] # this is the index of the rows to be deleted
        index.extend(indexTarget)

    _poses = np.delete(_poses, index, axis=0)
    _labels = np.delete(_labels, index, axis=0)
    labelLetters = np.delete(labelLetters, index, axis=0)
   
    deletedCount =  np.sum(np.array(index)>0)    
    print("Deleted count: ", deletedCount)

    return _poses, _labels, labelLetters

def filterLabelPoses(poses, labels, labels2filter):
   
    # this filters rows in poses for the label letters
   
    _poses = np.copy(poses)
    _labels = np.copy(labels)
    labelLetters = np.array([_labels[i][0] for i in range(_labels.shape[0])])
   
    # filter all poses samples rows for the letters
 
    index = []
    for labelTarget in labels2filter:
        indexTarget = np.where(labelLetters==labelTarget)[0] # this is the index of the rows to be deleted
        index.extend(indexTarget)

    _poses = _poses[index]
    _labels = _labels[index]
    labelLetters = labelLetters[index]
   
    filteredCount =  np.sum(np.array(index)>0)    
    print("Filtered count: ", filteredCount)

    return _poses, _labels, labelLetters

def flipPoses(poses, flip='H'):
   
    ##
    ##       NOTE !!!  THIS ASSUMES THAT THE Z AXIS IS THE DEPTH AND X IS
    ##       HORIZONTAL AXIS AND Y IS VERTICAL AXIS !!!!
    ##
    # returns poses flipped horizontally or vertically - based on flip = 'H' or 'V'
    #
   
    ##
    ##  !!! WHEN WE CHANGE THE X AND Y VALUES OF THE KEYPOINTS BELOW, THE Z VALUES STAY WHICH IS GOOD
    ##      - SO THAT THE DEPTH VALUE STAYS THE SAME WITH THE FLIPPED POSE - SO OUR XZ PLANE VIEW WILL
    ##      REFLECT THE FLIP CORRECTLY WITH THE CORRECT Z DEPTH VALUES FOR THE NEW (FLIPPED) POSE.
    ##
   
    # for each pose in the poses
    for poseIndex, pose in enumerate(poses):
       
        # for each keypoint in a pose of 3x21 values
       
        # get the min max of x and y
        minX = np.min(pose[0::3])
        minY = np.min(pose[1::3])
        maxX = np.max(pose[0::3])
        maxY = np.max(pose[1::3])
        centroidX = (minX + maxX)/2
        centroidY = (minY + maxY)/2      
       
        # flip the pose X values for horizontal, Y values if vertical
        if flip == 'H':  # FLIP HORIZ MEANS X VALS CHANGE ACROSS THE MID X
            pose[0::3] = (2 * centroidX) - pose[0::3]  
        else:            # FLIP VERT MEANS Y VALS CHANGE ACROSS THE MID Y
            pose[1::3] = (2 * centroidY) - pose[1::3]

    return poses

def plotPose(plt, pose, label, size, showJointCoords, showRotations, showPalmAlignments):
 
    # set up a 3d plot - size is in inches x inches
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111, projection='3d')
    if label is not None: ax.set_title(label)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # for each finger in hand
    for finger in FINGERS:
       
        colour = colours[finger]
           
        # for each bone in finger
        x = []; y = []; z = []; rot = []
        for bone in allBones[finger]:

            boneId, parentBone, head, tail = bone     # e.g. for index, bone2, we get (6,5,6,7)              
            headXYZ = getPoint(pose, head)  
            tailXYZ = getPoint(pose, tail)
            headX, headY, headZ = headXYZ
            tailX, tailY, tailZ = tailXYZ  
               
            if parentBone == -1:  # it's a palm bone                                
                boneDirection = getDirection(headXYZ, tailXYZ)                  
                boneRotation = boneDirection
                boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )                
            else:                
                parentBone, null, parentBonehead, parentBonetail = Bones[parentBone]                          
                parentHeadXYZ = getPoint(pose, parentBonehead)  
                parentTailXYZ = getPoint(pose, parentBonetail)
                                             
                # get bone euler x,y,z plane rotations - these are wrt the local coord system  
                boneDirection = getDirection(headXYZ, tailXYZ)  
                                                                                           
                # get parent bone euler x,y,z plane rotations - these are wrt the local coord system
                parentBoneDirection = getDirection(parentHeadXYZ, parentTailXYZ)
                   
                # get the bone euler rotation RELATIVE to the parent - IS INDEPENDENT OF LOCAL COORD SYSTEM !    
                boneRotation = tuple([boneDirection[i] - parentBoneDirection[i] for i in range(3)])
               
                # format it to fit onto the plot
                boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )
           
            #print(boneId, boneRotation) ## for debug  
       
            # set x, y and z values for the finger plot            
            x, y, z =  [headX, tailX], [headY, tailY], [headZ, tailZ]            
            rot =  [boneRotation] # np.append(rot, [boneRotation])          
               
            ax.plot(x,y,z, marker=finger, c=colour, label=fingers[finger])
           
            if showRotations == True:
               
                # draw in the bone rotation midway on the bone
                textContent = "Bone" + str(boneId) + " " +  str(rot)
                ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)
 
            if showPalmAlignments == True:
               
                # draw in the bone rotation midway on the bone
                if boneId == 8:
                    textContent = "Palm " + str(boneId) + " " +  str(rot)
                    ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)
               
            if  showJointCoords == True:    
               
                # draw in the keypoint coords at the joint keypoint
                headXYZ = tuple([(int(headX*100))/100, (int(headY*100))/100, (int(headZ*100))/100] )
                textContent = "Point " + str(head) + str(headXYZ)
                ax.text(x[0], y[0], z[0], textContent, fontsize=8, color=colour)
           
    #ax.legend(loc="best")    
    plt.show()
    print(label)          

    return
           
def plotPoses(plt, poses, labels, size, showJointCoords, showRotations, showPalmAlignments, delay):

    # set up a 3d plot - size is in inches x inches  
    #fig = plt.figure(num=1, figsize=(size,size))
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111, projection='3d')
   
    # for each hand pose in poses, draw in the plot
    for poseIndex, pose in enumerate(poses):
   
        ax.clear()  # clear the last pose off the plot      
       
        # for each finger in hand
        for finger in FINGERS:
       
            colour = colours[finger]
           
            # for each bone in finger
            x = []; y = []; z = []; rot = []
            for bone in allBones[finger]:

                boneId, parentBone, head, tail = bone     # e.g. for index, bone2, we get (6,5,6,7)                                
                headXYZ = getPoint(pose, head)  
                tailXYZ = getPoint(pose, tail)
                headX, headY, headZ = headXYZ
                tailX, tailY, tailZ = tailXYZ    
               
                if parentBone == -1:  # it's a palm bone                                
                    boneDirection = getDirection(headXYZ, tailXYZ)                  
                    boneRotation = boneDirection
                    boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )                
                else:                
                    parentBone, null, parentBonehead, parentBonetail = Bones[parentBone]                          
                    parentHeadXYZ = getPoint(pose, parentBonehead)  
                    parentTailXYZ = getPoint(pose, parentBonetail)
                                             
                    # get bone euler x,y,z plane rotations - these are wrt the local coord system  
                    boneDirection = getDirection(headXYZ, tailXYZ)  
                                                                                           
                    # get parent bone euler x,y,z plane rotations - these are wrt the local coord system  
                    parentBoneDirection = getDirection(parentHeadXYZ, parentTailXYZ)

                    # get the bone euler rotation RELATIVE to the parent - IS INDEPENDENT OF LOCAL COORD SYSTEM !
                    boneRotation = tuple([boneDirection[i] - parentBoneDirection[i] for i in range(3)])
               
                    # format it to fit onto the plot
                    boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )
               
                #print(boneId, boneRotation)    
       
                # set x, y and z values for the finger plot            
                x, y, z =  [headX, tailX], [headY, tailY], [headZ, tailZ]            
                rot =  [boneRotation] # np.append(rot, [boneRotation])          
               
                ax.plot(x,y,z, marker=finger, c=colour, label=fingers[finger])
           
                if showRotations == True:
                    # draw in the bone rotation midway on the bone
                    textContent = "Bone" + str(boneId) + " " +  str(rot)
                    ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)

                if showPalmAlignments == True:
                    # draw in the bone rotation midway on the bone
                    if boneId == 8:
                        textContent = "Palm " + str(boneId) + " " +  str(rot)
                        ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)

                if showJointCoords == True:    
                    # draw in the keypoint coords at the joint keypoint
                    headXYZ = tuple([(int(headX*100))/100, (int(headY*100))/100, (int(headZ*100))/100] )
                    textContent = "Point " + str(head) + str(headXYZ)
                    ax.text(x[0], y[0], z[0], textContent, fontsize=8, color=colour)
           
        #ax.legend(loc="best")  
        ax.set_xlabel("X Axis", fontsize=8)
        ax.set_ylabel("Y Axis", fontsize=8)
        ax.set_zlabel("Z Axis", fontsize=8)
        ax.set_title("Pose " + str(poseIndex) + " :  " + labels[poseIndex])
        plt.show() #block=True)
        plt.pause(delay)
   
    return

def plotFingerPoses(plt, poses, labels, size, showJointCoords, showRotations, showPalmAlignments, delay, fingerIDs):

    # set up a 3d plot - size is in inches x inches  
    #fig = plt.figure(num=1, figsize=(size,size))
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111, projection='3d')
   
    # for each hand pose in poses, draw in the plot
    for poseIndex, pose in enumerate(poses):
   
        ax.clear()  # clear the last pose off the plot      
       
        # get the wrist coords
        #wristXYZ = getPoint(pose, 0)    
        #wristX, wristY, wristZ = wrist
        #w = [wristX, wristY]

        # for each finger in hand
        for finger in fingerIDs:  #FINGERS:
       
            colour = colours[finger]
           
            # for each bone in finger
            x = []; y = []; z = []; rot = []
            for bone in allBones[finger]:

                boneId, parentBone, head, tail = bone     # e.g. for index, bone2, we get (6,5,6,7)                                
                headXYZ = getPoint(pose, head)  
                tailXYZ = getPoint(pose, tail)
                headX, headY, headZ = headXYZ
                tailX, tailY, tailZ = tailXYZ    
               
                if parentBone == -1:  # it's a palm bone                                
                    boneDirection = getDirection(headXYZ, tailXYZ)                  
                    boneRotation = boneDirection
                    boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )                
                else:                
                    parentBone, null, parentBonehead, parentBonetail = Bones[parentBone]                          
                    parentHeadXYZ = getPoint(pose, parentBonehead)  
                    parentTailXYZ = getPoint(pose, parentBonetail)
                                             
                    # get bone euler x,y,z plane rotations - these are wrt the local coord system  
                    boneDirection = getDirection(headXYZ, tailXYZ)  
                                                                                           
                    # get parent bone euler x,y,z plane rotations - these are wrt the local coord system  
                    parentBoneDirection = getDirection(parentHeadXYZ, parentTailXYZ)

                    # get the bone euler rotation RELATIVE to the parent - IS INDEPENDENT OF LOCAL COORD SYSTEM !
                    boneRotation = tuple([boneDirection[i] - parentBoneDirection[i] for i in range(3)])
               
                    # format it to fit onto the plot
                    boneRotation = tuple([ (int(boneRotation[i]*100)) /100 for i in range(3)] )
               
                #print(boneId, boneRotation)    
       
                # set x, y and z values for the finger plot            
                x, y, z =  [headX, tailX], [headY, tailY], [headZ, tailZ]            
                rot =  [boneRotation] # np.append(rot, [boneRotation])          
               
                ax.plot(x,y,z, marker=finger, c=colour, label=fingers[finger])
           
                if showRotations == True:
                    # draw in the bone rotation midway on the bone
                    textContent = "Bone" + str(boneId) + " " +  str(rot)
                    ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)

                if showPalmAlignments == True:
                    # draw in the bone rotation midway on the bone
                    if boneId == 8:
                        textContent = "Palm " + str(boneId) + " " +  str(rot)
                        ax.text( (x[1]+x[0])/2, (y[1]+y[0])/2, (z[1]+z[0])/2, textContent, fontsize=8, color=colour)

                if showJointCoords == True:    
                    # draw in the keypoint coords at the joint keypoint
                    headXYZ = tuple([(int(headX*100))/100, (int(headY*100))/100, (int(headZ*100))/100] )
                    textContent = "Point " + str(head) + str(headXYZ)
                    ax.text(x[0], y[0], z[0], textContent, fontsize=8, color=colour)
           
        #ax.legend(loc="best")  
        ax.set_xlabel("X Axis", fontsize=8)
        ax.set_ylabel("Y Axis", fontsize=8)
        ax.set_zlabel("Z Axis", fontsize=8)
        ax.set_title("Pose " + str(poseIndex) + " :  " + labels[poseIndex])
        plt.show() #block=True)
        plt.pause(delay)
   
    return

def rotateAndScalePoses(poses, config):  

    ## THIS USES CONFIG(HANDSCALE) SO SET TO 1 FOR UNITSCALING
   
    #
    #   Iterate through all the poses and add rotation and scaling variations.
    #   and returns a unit pose
   
    _poses = np.copy(poses)
    for poseIndex, pose in enumerate(_poses):  
       
        #
        #  First, rotate the pose to get random pose rotation variation in the dataset. The random rotation
        #  in each of the 3 axes is within the max limit in the config. Rotate the pose about the wrist.
        #  
       
        wrist_XYZ = getPoint(pose, WRIST)
        rotation_X, rotation_Y, rotation_Z = 0, 0, 0
        if config["rotationMin"] != 0 or config["rotationMax"] != 0:
            rotation_X = random.randint(config["rotationMin"], config["rotationMax"])
            rotation_Y = random.randint(config["rotationMin"], config["rotationMax"])
            rotation_Z = random.randint(config["rotationMin"], config["rotationMax"])
            pose = rotatePose(wrist_XYZ, pose, (rotation_X, rotation_Y, rotation_Z))

        #
        #   Scale the pose coord values and fit them into the canvas limits.
        #
       
        # Find the min and max X and Y coord of the pose values. This is the bounding box within which the
        # pose hand will fit. (Use np slicing. [;, iterates through all rows, 0::3] takes every 3rd value
        # from 1st col for Xs, then 1::3 for Ys)
        minX, minY, maxX, maxY = np.min(pose[0::3]), np.min(pose[1::3]), np.max(pose[0::3]), np.max(pose[1::3])

        # The bounding box is the maxmin differences across all points. This is in Blender coord units(metres).
        # Set the bounding box to be a square and is the max of H and W.
        boundingBoxWH = max(abs(maxX - minX), abs(maxY - minY))
           
        # Scale the pose point cloud by scaleX and scaleY by reducing boundingbox.
        ## DONT SCALE HERE - BECAUSE IT ONLY SCALES EITHER XY OR YZ BUT IM IN 3D WORLD HERE!!!
        ## SO IT ONLY MAKES SENSE TO PALMSCALE IE CHANGE THE RELATIVE LENGHT OF THE PALM BONES.
        boundingBoxW = boundingBoxWH * (1 / config["handScale_X"])
        boundingBoxH = boundingBoxWH * (1 / config["handScale_Y"])
       
        # Move the bounding box so that the top left corner (minX,minY) is at image pixel origin 0,0 by
        # shifting the X and Y coords of all the poses.
        pose[0::3] = pose[0::3] - minX
        pose[1::3] = pose[1::3] - minY

        # Scale the box to be a unit size              #   (not the canvas size)
        pose[0::3] = (pose[0::3] / boundingBoxW)       #   * canvas[0]    
        pose[1::3] = (pose[1::3] / boundingBoxH)       #   * canvas[1]              
       
    return _poses