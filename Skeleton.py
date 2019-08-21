import copy
import cv2
import numpy as np
from PIL import Image, ImageDraw


class Skeleton(object):
    """
    Class that represents the skeleton information.

    Construct a Skeleton object by passing the skeleton array with shape (180).
    """

    def __init__(self,data):
        """
        Constructor. Reads skeleton information from given raw data.
        """
        self.rawData = np.array(data, dtype=np.float32)
        # Create an object from raw data
        self.joints=dict();
        pos=0
        self.joints['HipCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['Spine']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['ShoulderCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['Head']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['ShoulderLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['ElbowLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['WristLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['HandLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['ShoulderRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['ElbowRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['WristRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['HandRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['HipLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['KneeLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['AnkleLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['FootLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['HipRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['KneeRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['AnkleRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos += 9
        self.joints['FootRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))

    def getRawData(self):
        """
        Returns all the information as numpy array which can be used to construct a skeleton object later.
        """
        return self.rawData

    def getAllData(self):
        """
        Return a dictionary with all the information for each skeleton node.
        """
        return self.joints

    def getWorldCoordinates(self):
        """
        Get World coordinates for each skeleton node.
        """
        skel=dict()
        for key in list(self.joints.keys()):
            skel[key]=self.joints[key][0]
        return skel

    def getJoinOrientations(self):
        """
        Get orientations of all skeleton nodes.
        """
        skel=dict()
        for key in list(self.joints.keys()):
            skel[key]=self.joints[key][1]
        return skel

    def getPixelCoordinates(self):
        """
        Get Pixel coordinates for each skeleton node.
        """
        skel=dict()
        for key in list(self.joints.keys()):
            skel[key]=self.joints[key][2]
        return skel

    def resizePixelCoordinates(self, cropH=[10,330], cropW=[140,460], scale=0.25):
        """
        Resize joint pixel coordinates in-place. Default values correspond to
        configuration of dataset. If you want to match the skeleton with images
        then use the default parameters.

        @param cropH: list of indices defining the crop area for the height axis
            (fetch pixels between the first and the second indices).
        @param cropW: list of indices defining the crop area for the width axis
            (fetch pixels between the first and the second indices).
        @param scale: amount of scale after cropping.
        """
        for key in list(self.joints.keys()):
            c = self.joints[key][2]
            # Height
            c[1] = int((c[1] - cropH[0])*scale)
            # Width
            c[0] = int((c[0] - cropW[0])*scale)

    def toImage(self,width,height,bgColor=(255,255,255)):
        """
        Create an image for the skeleton information.
        @param width: width of the corresponding image.
        @param height: height of the corresponding image.
        @param width: background color. (255,255,255) for white.
        """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        # Draw bones.
        for link in SkeletonConnectionMap:
            p=copy.copy(self.getPixelCoordinates()[link[1]])
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=3)
        # Draw joints.
        for node in list(self.getPixelCoordinates().keys()):
            p=self.getPixelCoordinates()[node]
            r=2
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = np.array(im)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
