# RGB Vegetation Indices
# BoMeyering 2024

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def clahe_channel(img: np.ndarray, clip_limit=100) -> np.ndarray:
    """
    Performs CLAHE on the l* channel of an image

    Args:
        img (np.ndarray): a three channel image in opencv format (BGR order)

    Returns:
        np.ndarray: The adjusted image
    """
    # Convert to lab
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Apply CLAHE
    l_channel = lab_img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    new_l = clahe.apply(l_channel)
    lab_img[:, :, 0] = new_l
    
    # Convert back to BGR
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    return bgr_img

class VegIndex:
    """VegIndices: All indices get returned on a 0-1 (or very close!) scale. Normalize to 0-255 by (index*255).astype(np.uint8)"""

    def __init__(self, img):
        if len(img.shape) < 3:
            raise ValueError("'img' should be have a channel dimension.")
        if img.shape[2] != 3:
            raise ValueError("'img' should have 3 channels in dimension 2.")
        self.img = np.moveaxis(img, source=2, destination=0).astype(float)
        self.eps = 0.000001

    def green_blue(self):
        """Green minus blue index"""
        B, G, R = self.img

        index = G - B
        index = (index + 255) / 510

        return index
    
    def red_green(self):
        """Red minus green index"""
        B, G, R = self.img

        index = R - G
        index = (index + 255) / 510

        return index

    def excess_green(self):
        """Excess green index"""
        B, G, R = self.img

        index = 2 * G - R - B
        index = (index + 510) / 1020

        return index

    def exg_exr(self):
        """Excess green minus excess red index"""
        B, G, R = self.img

        exg = 2 * G - R - B
        exr = 1.4 * R - G

        index = exg - exr
        index = (index + 867) / 1632

        return index

    def ndi(self):
        """Normalized difference index"""
        B, G, R = self.img

        index = (G - R) / np.clip((G + R), a_min=1, a_max=510)
        index = (index + 1) / 2

        return index

    def woebbecke_index(self):
        """Woebbecke index"""
        B, G, R = self.img

        index = (G - B) / np.clip(np.abs(R - G), a_min=1, a_max=255)
        index = (index + 255) / 510

        return index

    def cive(self):
        """Color index of vegetation extraction"""
        B, G, R = self.img

        index = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
        index = (index + 188.01755) / 417.43505

        return index.astype(np.uint8)

    def cive_inv(self):
        """Color Index of Vegetation Extraction Inverse"""

        B, G, R = self.img

        index = -0.441 * R + 0.811 * G - 0.385 * B - 18.78745
        index = (index + 229.41745) / 417.43505

        return index

    def veg(self):
        """Veg Index"""
        B, G, R = self.img

        index = G / np.clip(((R**0.667) * (B**0.333)), a_min=1, a_max=255)
        index = index / 255

        return index
    
    def mgrvi(self):
        """
        Modified green red vegetation index
        Bendig et al 2015
        https://www.sciencedirect.com/science/article/pii/S0303243415000446
        """

        B, G, R = self.img

        index = (G**2 - R**2)/np.clip((G**2 + R**2), a_min=1, a_max=130050)
        index = (index + 1)/2

        return index
    
    def rgbvi(self):
        """Red Green Blue Vegetation Index"""

        B, G, R = self.img

        index = (G**2 - (B*R))/np.clip(G**2 + (B*R), a_min=1, a_max=130050)
        index = (index + 1)/2

        return index

    def gli(self):
        """
        Green Leaf Index
        Louhaichi et al.
        https://www.tandfonline.com/doi/abs/10.1080/10106040108542184
        """

        B, G, R = self.img

        index = (2*G - R - B)/np.clip((2*G + R + B), a_min=1, a_max=1020)
        index = (index + 1)/2

        return index

    def rgri(self):
        """
        Red Green Ratio Index
        """

        B, G, R = self.img

        index = R/np.clip(G, a_min=1, a_max=255)
        # index = (index)/(255)*(255)

        return index.astype(np.uint8)

    def ngrdi(self):
        """
        Normalized Green minus Red Vegetation Index
        Compton Tucker 1979
        https://www.sciencedirect.com/science/article/pii/0034425779900130
        """

        B, G, R = self.img

        index = (G - R)/np.clip(G + R, a_min=1, a_max=510)
        index = (index + 1)/(1 + 1)*(255) - 1

        return index.astype(np.uint8)

    def ngbdi(self):
        """
        Normalized Green Blue Difference Index
        Du and Noguchi, 2017
        https://www.mdpi.com/2072-4292/9/3/289
        """

        B, G, R = self.img

        index = (G - B)/np.clip(G + B, a_min=1, a_max=510)
        index = (index + 1)/2

        return index.astype(np.uint8)

    def vari(self):
        """
        Visible Atmospherically Resistant Vegetation Index
        Gitelson et al.
        
        """
        B, G, R = self.img

        index = (G - R)/(G + R - B + self.eps)
        index = np.clip(index, -1, 1)
        index = (index + 1)/2

        return index

    def kawashima(self):
        """
        Kawashima index
        Kawashima and Nakatani
        https://www.sciencedirect.com/science/article/abs/pii/S0305736497905448
        """

        B, G, R = self.img

        index = (R - B)/np.clip(R + B, a_min=1, a_max=510)
        index = (index + 1)/(1 + 1)*(255) + 1

        return index.astype(np.uint8)
    
    def mexg(self):
        """
        Modified Excess Green Index
        Burgos-Artizzu et al.
        https://www.sciencedirect.com/science/article/pii/S0168169910002620
        """

        B, G, R = self.img

        index = 1.262*G - 0.884*R - 0.311*B
        index = (index + 304.725)/(321.3 + 304.725)

        return index
