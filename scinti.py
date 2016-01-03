# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:58:49 2015

@author: sengery
"""
import numpy as np
import matplotlib.pyplot as plt

class scintigram:

    def __init__(self,size=256):
        self.image = np.zeros((size,size),dtype=int)
        self.size = size

    def get_center(self,center):
        """
        Berechnet aus in Aufgabenstellung gegebenen Bildkoordinaten Array-
        Indizes.
        """
        return np.array(center)*np.array([1,-1]) + self.size/2

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

    def show(self,fig=1,data=None,color="gray"):
        """
        Einfache Möglichkeit, die erzeugten Bilder und Daten auszugeben.
        """
        if np.all(data) == None:
            data = self.image
        plt.figure(fig)
        plt.imshow(data,cmap=color)

    def profile(self,data=None,y=0,fig=1,color="#cc0000",offset=0):
        """
        Legt ein Profil bei der angegebenen y-Koordinate durch das gesamte
        Bild. Kann entweder in neuer oder über vorhandener Figure ausgegeben
        werden.
        """
        if np.all(data) == None:
            data = self.image
        self.show(fig,data)
        plt.plot(np.arange(self.size),-data[-y+self.size/2,:]-offset+self.size,color=color)

    def histogram(self,data=None,fig=1,color="#ace600"):
        """
        Erzeugt ein Histogram der in den Daten vorhandenen Grauwerte. Kann auf
        Wunsch wiederum im analysierten Bild angegeben werden.
        """
        if np.all(data) == None:
            data = self.image
        self.show(fig,data)
        self.histo, self.bins = np.histogram(data,bins=np.arange(1,self.size,1))
        self.histo = self.histo.astype(float)/np.max(self.histo) * self.size
        plt.bar(self.bins[:-1]-1,-self.histo,bottom=self.size,width=1,color=color)

    def mean_information(self,fig):
        pass

    def difference(self,data=None,fig=2,color="gray"):
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

    def fft2(self,data=None,fig=5):
        """
        Erzeugt die 2D-Fouriertransformierte von data. Wird unter self.ft_image
        gespeichert. Gibt die Transformierte als Absolutbild aus.
        """
        if np.all(data) == None:
            data = self.image
        self.ft_image = np.fft.fftshift(np.fft.fft2(data))
        self.show(fig,np.abs(self.ft_image))

if __name__ ==   "__main__":
    pic = scintigram()
    pic.striped_square((60,60),100,100,200)
    pic.striped_square()
    pic.circle()
    pic.triangle()
    pic.normalize()

    pic.profile(y=-60)
    pic.histogram(fig=2)

    pic.difference(fig=3)
    pic.profile(pic.diff_image,y=60,offset=128,fig=3)
    pic.histogram(pic.diff_image,fig=4)
    pic.fft2(fig=5)
