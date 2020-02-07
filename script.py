#!/usr/bin/env python

''' This is a sample for histogram plotting for RGB images and grayscale images for better understanding of colour distribution

Benefit : Learn how to draw histogram of images
          Get familier with cv.calcHist, cv.equalizeHist,cv.normalize and some drawing functions

Level : Beginner or Intermediate

Functions : 1) hist_curve : returns histogram of an image drawn as curves
            2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Abid Rahman 3/14/12 debug Gary Bradski
'''

# Python 2/3 compatibility
from __future__ import print_function

from matplotlib.pyplot import *
import imageio

import numpy as np
import cv2 as cv

bins = np.arange(256).reshape(256,1)	#np.arange(256) --> [0,1,...,255]   | np.arange(256).reshape(256,1)  --> [[0],[1],...,[255]]

def hist_profil(ligne):
    h = np.zeros((300,256,3))
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    #color = [(0,0,255)]
    for ch,col in enumerate(color):
       hist_item = cv.calcHist([ligne],[ch],None,[256],[0,256])  # Calculates the histogram
       cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX) # Normalize the value to fall below 255, to fit in image 'h'
       hist=np.int32(np.around(hist_item))                      
       pts = np.column_stack((bins,hist))                       # stack bins and hist, ie [[0,h0],[1,h1]....,[255,h255]]
       cv.polylines(h,[pts],False,col)
    h=np.flipud(h) 
    return h

def hist_curve(img):
    h = np.zeros((300,256,3))
    if len(img.shape) == 2:			#si l'image est 2D
        color = [(255,255,255)]
    elif img.shape[2] == 3:			#si l'image est 3D
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):				#enumarate(liste) renvoie (indice, elementListe)
        hist_item = cv.calcHist([img],[ch],None,[256],[0,256])	#calcul de Histogramme
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(img):
    h = np.zeros((300,256,3))
    if len(img.shape)!=2:
        print("Applicable pour les images en niveau de gris uniquement")
        #print("so converting image to grayscale for representation"
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    hist_item = cv.calcHist([img],[0],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def main():
    import sys

    if len(sys.argv)>1:
        fname = sys.argv[1]
    else :
        fname = 'images/coul1.jpg'
        print("usage : python script.py [fichier_image]")

    im = cv.imread(cv.samples.findFile(fname))

    if im is None:
        print('Erreur de chargement du fichier:', fname)
        sys.exit(1)

    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)


    print(''' TP 1 \n
    MENU :\n
    a - Histogramme image couleur \n
    b - Profil d'intensite \n
    c - Histogramme RGB -> niveau de gris \n
    d - Histogram Equalized\n
    e - Histogram MIN-MAX \n
    f - Histogram CLAHE (niveau de gris) \n
    Esc - exit \n
    ''')

    cv.imshow('image',im)
    while True:
        k = cv.waitKey(0)
        if k == ord('b'):
            num = input('Numero de la ligne :')
            num = int(num)
            im_clone = im
            ligne = im_clone[num:,:]
            cv.line(im_clone, (0, num), (im_clone.shape[1], num), (0, 0, 255), 1)
  
            # find frequency of pixels in range 0-255 
            #histr = cv.calcHist([ligne],[0],None,[256],[0,256]) 
  
            hist = hist_curve(ligne)
            #hist_1col = hist_lines(ligne)
            cv.imwrite('hist_lignes.jpg', hist)
            cv.imwrite('profil.jpg', im_clone)
            #cv.imshow('Profil Intensite gray',hist_1col)
            cv.imshow('Profil Intensite',hist)
            # show the plotting graph of an image 
            cv.imshow('image',im_clone)
            #plot(histr) 
            #show()
            print('b')
        if k == ord('a'):
            curve = hist_curve(im)
            cv.imshow('image',im)
            cv.imshow('histogram',curve)
            print('a')
        elif k == ord('c'):
            print('c')
            lines = hist_lines(im)
            cv.imshow('image',gray)
            cv.imshow('histogram gray',lines)
        elif k == ord('d'):
            print('d')
            equ = cv.equalizeHist(gray)
            lines = hist_lines(equ)
            cv.imshow('image',equ)
            cv.imshow('Equalized',lines)
        elif k == ord('e'):
            print('e')
            norm = cv.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv.NORM_MINMAX)
            lines = hist_lines(norm)
            cv.imshow('image',norm)
            cv.imshow('MIN-MAX',lines)
        elif k == ord('f'):
            # create a CLAHE object (Arguments are optional).
             clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
             cl1 = clahe.apply(gray)
             lines = hist_lines(cl1)
             cv.imshow('IMAGE', cl1)
             cv.imshow('CLAHE',lines)
             print('f')           
        elif k == 27:
            print('ESC')
            cv.destroyAllWindows()
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
