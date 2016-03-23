# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:58:49 2015

@author: sengery
"""
import sys
import numpy as np

import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sip


class scintigram:

    def __init__(self,size=256.):
        self.image = np.zeros((size,size),dtype=int)
        self.size = size

        self.circle()
        self.triangle()
        self.striped_square()
        self.striped_square((60,60),thick=100,mean1=200)
        self.normalize()

    def get_center(self,center, size=256):
        """
        Berechnet aus in Aufgabenstellung gegebenen Bildkoordinaten Array-
        Indizes.
        """
        return np.array(center)*np.array([1,-1]) + size/2

    def normalize(self):
        """
        Da durch die zufällige Verteilung auch Werte über 255 entstehen können,
        kann mit dieser Funktion eine Normalisierung auf 255 vorgenommen werden.
        """
        self.image = self.image/float(np.max(self.image))
        self.image *= 255

    def striped_square(self,center=(-60,60),edge=100,thick=5,mean1=250,mean2=300):
        """
        Erzeugt ein Quadrat mit gegebener Kantenlänge um das Zentrum herum.
        Es können Streifen unterschiedlicher Flächendichte vorgegeben werden.
        """
        c_xy = self.get_center(center)
        for num in range(0,edge/thick):
            if not num%2:
                self.image[c_xy[1]-edge/2:c_xy[1]+edge/2,
                c_xy[0]-edge/2+num*thick:c_xy[0]-edge/2+(num+1)*thick] = \
                np.random.poisson(mean1,(edge,thick))
            elif num%2:
                self.image[c_xy[1]-edge/2:c_xy[1]+edge/2,
                c_xy[0]-edge/2+num*thick:c_xy[0]-edge/2+(num+1)*thick] = \
                np.random.poisson(mean2,(edge,thick))

    def circle(self,center=(-60,-60),radius=50,mean=50):
        """
        Erzeugt Kreis um gegebenes Zentrum.
        """
        c_xy = self.get_center(center)
        mask = np.fromfunction(lambda x,y: (c_xy[1]-x)**2 + (c_xy[0]-y)**2,
                               self.image.shape) < radius**2
        self.image[mask] = np.random.poisson(mean,self.image.shape)[mask]

    def triangle(self,top=(60,-10),height=100,mean=100):
        """
        Erzeugt ein Dreieck unterhalb des gegebenen Punktes.
        """
        top = self.get_center(top)
        side = height/np.sin(60/360. * 2 * np.pi)
        mask = np.fromfunction(lambda x,y: (np.abs(top[0]-y)<=(
        np.abs(top[1]-x)/height*side/2))*((top[1]-x) <= 0)*
        ((top[1]-x) >= -height),self.image.shape)
        self.image[mask] = np.random.poisson(mean,self.image.shape)[mask]



class evaluation:

    def __init__(self,scintigram):
        self.size = scintigram.size
        self.image = scintigram.image

    def show(self,fig=1,data=None,color="gray"):
        """
        Einfache Möglichkeit, die erzeugten Bilder und Daten auszugeben.
        """
        if np.all(data) == None:
            data = self.image
        plt.figure(fig)
        plt.imshow(data,cmap=color)
        plt.axis("off")
        plt.show()



    def profile(self,data=None,y=0,fig=1,color="#cc0000"):
        """
        Legt ein Profil bei der angegebenen y-Koordinate durch das gesamte
        Bild. Ausgabe: Bild mit eingezeichneter Lage des Profils und Verlauf
        der Grauwerte im Profil.
        """
        if np.all(data) == None:
            data = 1*self.image

        self.show(fig,data)
        profile_location = np.ones(self.size)*(self.size/2 - y)
        plt.plot(np.arange(256),profile_location, color=color)

        plt.figure(fig+1)
        plt.plot(np.arange(self.size),data[-y+self.size/2,:],color=color)

    def histogram(self,data=None,fig=3,exclude_zero=False,color="#ace600"):
        """
        Erzeugt ein Histogram der in den Daten vorhandenen Grauwerte. Ausgabe
        in beliebiger Figure. Per exclude_zero=True kann die Berechnung des
        Histograms auf Werte > 0 beschränkt werden, um eine detailliertere
        Darstellung des variablen Wertebereichs zu erhalten.
        """
        if np.all(data) == None:
            data = self.image

        histo, bins = np.histogram(data,bins=self.size-1)
        plt.figure(fig)

        if exclude_zero:
            bins = bins[1::]
            histo = histo[1::]

        plt.bar(bins[:-1],histo,width=1,color=color)

    def mean_skew(self,data=None):
        """
        Berechnet Mittelwert und Schiefe des Grauwerthistogramms.
        """
        if np.all(data) == None:
            data = self.image

        histo, bins = np.histogram(data,bins=255,normed=True)
        mean = np.sum(histo * bins)
        skew = (bins - mean)**3*bins

        return mean,skew

    def mean_information(self,data=None):
        """
        Berechnet die mittlere Information je Pixel für die gegebenen Daten.
        """
        if np.all(data) == None:
            data = self.image
        histo, bins = np.histogram(data,bins=self.size-1,normed=True)
        histo = np.sum(-1*histo[histo > 0]*np.log2(histo[histo>0]))
        return np.round(histo,3)

    def bit_layer(self,layer=0,data=None,plot=False):
        if np.all(data) == None:
            data = self.image
        bitdata = np.unpackbits(data.astype(np.uint8)).reshape(self.size,-1)
        if plot == True:
            self.plot_bitlayer(data)
        return bitdata[:,7-layer::8]

    def difference(self,data=None,fig=1,color="gray"):
        """
        Erzeugt ein zeilenweises Differenzbild.
        """
        if np.all(data) == None:
            data = self.image
        self.diff_image = np.zeros(data.shape)
        self.diff_image[:,1:] = data[:,:-1]
        self.diff_image = data - self.diff_image
        plt.figure(fig)
        self.show(fig,self.diff_image,color)

    def fft2(self,data=None,fig=1):
        """
        Erzeugt die 2D-Fouriertransformierte von data. Wird unter self.ft_image
        gespeichert. Gibt die Transformierte als Absolutbild aus.
        """
        if np.all(data) == None:
            data = self.image
        self.ft_image = np.fft.fftshift(np.fft.fft2(data))
        self.show(fig,np.abs(self.ft_image)**2)

    def rotate(self,data,angle):
        """
        Rotiert das Array data um den Winkel angle.
        """
        return sip.rotate(data,angle)

    def lowpass(self,data):
        """
        Wendet einen Tiefpass auf die übergebenen Daten an. Cutoff ist bei
        """
        ft_data = np.fft.fftshift(np.fft.fft2(data))
        lowpass = np.fromfunction(lambda x,y:np.sqrt((x-self.size/2)**2+
            (y-self.size/2)**2),data.shape)
        lowpass = np.array(lowpass < self.size/4,dtype=int)
        return np.fft.ifft2(np.fft.ifftshift(ft_data*lowpass))

    def bandpass(self,data):
        ft_data = np.fft.fftshift(np.fft.fft2(data))
        bandpass = np.fromfunction(lambda x,y:np.sqrt((x-self.size/2)**2+
            (y-self.size/2)**2),data.shape)
        bandpass = np.array((bandpass > self.size * 3./16)*
            (bandpass < self.size * 5./16),dtype=int)
        return np.fft.ifft2(np.fft.ifftshift(ft_data*bandpass))

    def highpass(self,data):
        ft_data = np.fft.fftshift(np.fft.fft2(data))
        highpass = np.fromfunction(lambda x,y:np.sqrt((x-self.size/2)**2+
            (y-self.size/2)**2),data.shape)
        highpass = np.array(highpass > self.size*3./8,dtype=int)
        return np.fft.ifft2(np.fft.ifftshift(ft_data*highpass))

    def shear(self):
        image = self.image
        rows,cols = self.image.shape
        shear_img = np.zeros(self.image.shape)
        for x in xrange(1,cols-1,1):
            for y in xrange(1,rows-1,1):
                pass
        return shear_img


if __name__ == "__main__":
    plt.close("all")
    pic = scintigram()
    ev = evaluation(pic)

    ev.profile(y=60)
    ev.histogram(exclude_zero=False)
