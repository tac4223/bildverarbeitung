# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:58:49 2015

@author: sengery
"""
import numpy as np

import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sip


class scintigram:

    def __init__(self,size=256.):
        self.image = np.zeros((size,size),dtype=int)
        self.size = size
        self.diff_image = None

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

class grayscale:
    """
    Erstellt einen quadratischen Graukeil mit 256 Grauwerten.
    """
    def __init__(self,size=256):
        self.size = size
        self.image = np.ones((size,size)) * np.linspace(0,size,256).astype(int)

    def inverse(self):
        """
        Wendet inverse Kennlinie auf Bild an.
        """
        return 255 - self.image

    def squared(self):
        """
        Wendet quadratische Kennlinie an.
        """
        return np.round(self.image**2 / 255.,0)

    def root(self):
        """
        Wendet die Wurzelfunktion als Kennlinie an.
        """
        return np.round(np.sqrt(self.image*255),0)

    def binary(self,lower,upper):
        """
        Binarisiert das Bild, wobei alle Werte zwischen lower und upper als 1
        gesetzt werden.
        """
        return (self.image > lower) * (self.image < upper)

    def gaussian(self, sigma=85.,mu=0.):
        """
        Gauß-Kennlinie.
        """
        return np.round(258 - 54942/(np.sqrt(2*np.pi)*sigma) * np.exp(-(self.image - mu)**2/(2*sigma**2)),0)


class tools:

    @classmethod
    def show(self,image,fig=1,color="gray"):
        """
        Einfache Möglichkeit, die erzeugten Bilder und Daten auszugeben.
        """
        plt.figure(fig)
        plt.imshow(image,cmap=color)
        plt.axis("off")
        plt.show()


    @classmethod
    def profile(self,image,y=0,fig=1,color="#cc0000"):
        """
        Legt ein Profil bei der angegebenen y-Koordinate durch das gesamte
        Bild. Ausgabe: Bild mit eingezeichneter Lage des Profils und Verlauf
        der Grauwerte im Profil.
        """
        self.show(image,fig)

        height,width = image.shape

        profile_location = np.ones(width)*(height/2 - y)
        plt.plot(np.arange(256),profile_location, color=color)

        plt.figure(fig+1)
        plt.plot(np.arange(width),image[-y+height/2,:],color=color)

    @classmethod
    def histogram(self,image,fig=3,exclude_zero=False,color="#ace600",bin_count=255):
        """
        Erzeugt ein Histogram der in den Daten vorhandenen Grauwerte. Ausgabe
        in beliebiger Figure. Per exclude_zero=True kann die Berechnung des
        Histograms auf Werte > 0 beschränkt werden, um eine detailliertere
        Darstellung des variablen Wertebereichs zu erhalten.
        """
        histo, bins = np.histogram(image,bins=bin_count)
        plt.figure(fig)

        if exclude_zero:
            bins = bins[1::]
            histo = histo[1::]

        plt.bar(bins[:-1],histo,width=1,color=color)

    @classmethod
    def mean_skew(self,image,exlude_zero=False,bin_count=255):
        """
        Berechnet Mittelwert und Schiefe des Grauwerthistogramms.
        """
        histo, bins = np.histogram(image,bins=bin_count,normed=True)
        bins = bins[:-1]

        if exlude_zero:
            histo = histo[1:]
            bins = bins[1:]

        mean = np.sum(histo * bins)
        skew = np.sum((bins - mean)**3*histo)

        return mean,skew

    @classmethod
    def mean_information(self,image,bin_count=255):
        """
        Berechnet die mittlere Information je Pixel für die gegebenen Daten.
        """
        histo, bins = np.histogram(image,bins=bin_count,normed=True)
        histo = np.sum(-1*histo[histo > 0]*np.log2(histo[histo>0]))
        return np.round(histo,3)

    @classmethod
    def bit_layer(self,image,layer=0):
        """
        Berechnet zu gegebenem Bild beliebige Bitebenen.
        """
        height,width = image.shape

        bitdata = np.unpackbits(image.astype(np.uint8)).reshape(height,-1)
        return bitdata[:,7-layer::8]

    @classmethod
    def difference(self,image,fig=1,color="gray"):
        """
        Erzeugt ein zeilenweises Differenzbild.
        """
        diff_image = np.zeros(image.shape)
        diff_image[:,1:] = image[:,:-1]
        diff_image = image - diff_image
        return diff_image

    @classmethod
    def fft2(self,image):
        """
        Erzeugt die 2D-Fouriertransformierte von data, fftshift ist in der
        Ausgabe bereits angewendet.
        """
        return np.fft.fftshift(np.fft.fft2(image))

    @classmethod
    def rotate(self,image,angle):
        """
        Rotiert das Array data um den Winkel angle.
        """
        return sip.rotate(image,angle)

    @classmethod
    def lowpass(self,image,cutoff=0.25):
        """
        Wendet einen Tiefpass auf die übergebenen Daten an. Cutoff ist variabel,
        wird noch mit 0.5 multipliziert um der Ausgabe von fft2 Rechnung zu
        tragen (Werte für Nyqvist-Frequenz liegen nach fftshift am Rand).
        """
        height,width = image.shape

        ft_data = np.fft.fftshift(np.fft.fft2(image))
        lowpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)
        lowpass = np.array(lowpass < height/(cutoff*0.5),dtype=int)

        return np.fft.ifft2(np.fft.ifftshift(ft_data*lowpass))

    @classmethod
    def bandpass(self,image,cutoff=[0.375,0.525]):
        """
        Bandpass, ansonsten identisch zu lowpass.
        """
        height,width = image.shape

        ft_data = np.fft.fftshift(np.fft.fft2(image))
        bandpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)

        bandpass = np.array((bandpass > height * cutoff[0]*0.5)*
            (bandpass < height * cutoff[1]*0.5),dtype=int)

        return np.fft.ifft2(np.fft.ifftshift(ft_data*bandpass))

    @classmethod
    def highpass(self,image,cutoff=0.75):
        """
        Hochpass, ansonsten identisch zu lowpass.
        """
        height,width = image.shape

        ft_data = np.fft.fftshift(np.fft.fft2(image))
        highpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)

        highpass = np.array(highpass > height*cutoff*.5,dtype=int)

        return np.fft.ifft2(np.fft.ifftshift(ft_data*highpass))

    @classmethod
    def shear(self,image):
        """
        Führt eine festgelegte Scherung auf die übergebene Bildmatrix aus.
        Die Bildgröße ändert sich durch die Scherung.
        """
        height,width = image.shape
        origin = np.mgrid[0:height,0:width]

        index = np.ones((3,height,width))
        index[0,:,:] = origin[1][::-1,:]
        index[1,:,:] = origin[0][::-1,:]
        #umgekehrte Zeilenfolge, da y-Achse im Vergleich zum Skript gespiegelt
        #ist.

        for x in np.arange(width):
            for y in np.arange(height):
                index[:,y,x] = np.dot(
                [[1,0.5*np.sqrt(2),0],[0,0.5*np.sqrt(2),0],[0,0,1]],
                  index[:,y,x])
       #Anwenden der Scherungsmatrix auf jedes Koordinatentripel in Index.

        x = index[0,:,:].astype(int)
        y = index[1,:,:].astype(int)

        shear = np.ones((np.max(y)+1,np.max(x)+1))
        shear[y,x] = image

        return shear[::-1,:]
        #Wiederum Reihenfolge umkehren damit Bild korrekt ausgegeben wird.

    @classmethod
    def neighbors_3x3(self,image):
        """
        Erzeugt ein Array das für jeden Pixel des input-Bildes alle 9 ELemente
        der 9er-Nachbarschaft enthält.
        Reihenfolge: (y,x),(y,x+1),(y-1,x+1),(y-1,x),(y-1,x-1),(y,x-1),
        (y+1,x-1),(y+1,x),(y+1,x+1) für neighbors[0] bis neighbors[8].

        Vorteil: Keine Schleifen, vermutlich schnell.
        Nachteil: 9facher Speicherplatz des Ausgangsbildes benötigt.
        """
        height,width = image.shape
        y,x = np.mgrid[1:height-1,1:width-1]
        y_pos,x_pos = y+1,x+1
        y_neg,x_neg = y-1,x-1

        neighbors = np.ones((9,height-2,width-2))
        neighbors[0,:,:] = image[1:-1,1:-1]
        neighbors[1,:,:] = image[y,x_pos]
        neighbors[2,:,:] = image[y_neg,x_pos]
        neighbors[3,:,:] = image[y_neg,x]
        neighbors[4,:,:] = image[y_neg,x_neg]
        neighbors[5,:,:] = image[y,x_neg]
        neighbors[6,:,:] = image[y_pos,x_neg]
        neighbors[7,:,:] = image[y_pos,x]
        neighbors[8,:,:] = image[y_pos,x_pos]

        return neighbors

    @classmethod
    def mean_filter(self,image):
        """
        Mittelwertsfilter für 3x3 Nachbarschaft.
        """
        return np.mean(self.neighbors_3x3(image),axis=0)

    @classmethod
    def median_filter(self,image):
        """
        Medianfilter für 3x3 Nachbarschaft
        """
        return np.median(self.neighbors_3x3(image),axis=0)

    @classmethod
    def gauss_filter(self,image):
        """
        Binomialfilter für 4er Nachbarschaft
        """
        gauss = self.neighbors_3x3(image)
        gauss[0] *= 4
        gauss[1:] *= 2
        return 1/16. * np.sum(gauss,axis=0)

    @classmethod
    def laplace_filter_8(self,image):
        """
        Laplace-Filter für 8er Nachbarschaft.
        """
        laplace = 1*self.neighbors_3x3(image)
        laplace[0] *= 8
        laplace[1:] *= -1
        return np.sum(laplace,axis=0)

    @classmethod
    def sobel_filter(self,image):
        """

        """
        neighbors = self.neighbors_3x3(image)
        sobel_x = neighbors[[1,2,4,5,6,8]]
        sobel_x[[0,3]] *= 2
        sobel_x[[2,3,4]] *= -1

        sobel_y = neighbors[[2,3,4,6,7,8]]
        sobel_y[[1,4]] *= 2
        sobel_y[[0,1,2]] *= -1

        return np.abs([np.sum(sobel_x,axis=0),np.sum(sobel_y,axis=0)])

    @classmethod
    def roberts_filter(self,image):
        neighbors = self.neighbors_3x3(image)

        roberts_x = neighbors[0] - neighbors[8]
        roberts_y = neighbors[1] - neighbors[7]

        return np.abs([roberts_x,roberts_y])

    @classmethod
    def image_moment(self,image,p,q):
        height,width = image.shape
        yx = np.mgrid[0:height,0:width]

        return np.sum(image*yx[0]**q*yx[1]**p)

    @classmethod
    def transition_matrix(self,image,delta):
        bins = np.unique(image)
        count = len(bins)

        height, width = image.shape
        working_image = np.ones((height+2,width+2))
        working_image *= -999
        working_image[1:-1,1:-1] = 1*image

        transition_matrix = np.zeros((count,count))

        for m in range(count):
            index = np.where(working_image==bins[m])
            for n in range(count):
                index = np.where(working_image==bins[m])
                transition_matrix[m,n] = np.sum(working_image[index[0]+delta[1],index[1]+delta[0]] == bins[n])

        return transition_matrix


if __name__ == "__main__":
    plt.close("all")
    pic = scintigram()

#    image = np.zeros((25,25))
#    image[10:15,10:20] = 1
#    tools.show(image)
#
#    m00 = tools.image_moment(image,0,0)
#    m01 = tools.image_moment(image,0,1)
#    m10 = tools.image_moment(image,1,0)
#
#    print(m10/m00)
#    print(m01/m00)