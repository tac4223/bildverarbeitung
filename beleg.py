# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:58:49 2015

@author: sengery
"""
import numpy as np

import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sip


class scintigram:

    def __init__(self,size=256):
        """
        Parameter
        -----------------------------------------------------------------------
        size : int
            Kantenlänge des erzeugten Bildes. Standardwert 256.

        Funktionen
        -----------------------------------------------------------------------
        get_center :
            Koordianten-Umrechnung.

        normalize :
            Beschränken der Grauwerte auf 255.

        striped_square :
            Erstellt gestreiftes oder homogenes Quadrat.

        circle :
            Erstellt Kreis.

        triangle :
            Erstellt Dreieck.

        Beschreibung
        -----------------------------------------------------------------------
        Erstellt standardmößig die in Aufgabe 1.1 geforderten geometrischen
        Formen mit entsprechenden zufälligen Grauwerten. Die Formen sind aber
        prinzipiell beliebig anzuordnen und zu skalieren.
        """
        self.image = np.zeros((size,size),dtype=int)
        self.size = size
        self.diff_image = None

        self.circle()
        self.triangle()
        self.striped_square()
        self.striped_square((60,60),thick=100,mean1=200)
        self.image = self.normalize(self.image)

    def get_center(self,center,size=256):
        """
        Parameter
        -----------------------------------------------------------------------
        center : array_like
            In der Form [x,y] oder [[x1,y1],[x2,y2],...]


        Beschreibung
        -----------------------------------------------------------------------
        Berechnet aus in Aufgabenstellung gegebenen Bildkoordinaten Array-
        Indizes.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Array mit den korrespondierenden Array-Indizes, in der Form
            [Spalte,Zeile]
        """
        return np.array(center)*np.array([1,-1]) + size/2

    @classmethod
    def normalize(self,image,low=0,high=255):
        """
        Beschreibung
        -----------------------------------------------------------------------
        Da durch die zufällige Verteilung auch Werte über 255 entstehen können,
        kann mit dieser Funktion eine Normalisierung auf 255 vorgenommen werden.

        Ausgabe
        -----------------------------------------------------------------------
        self.image wird auf 255 normalisiert, keine weitere Ausgabe.
        """
        image += low - np.min(image)
        image = image/float(np.max(image))
        image *= high
        return image

    def striped_square(self,center=(-60,60),edge=100,thick=5,mean1=250,
                       mean2=300):
        """
        Parameter
        -----------------------------------------------------------------------
        center : array_like
            Mittelpunkt des Quadrats in der Form [x,y], Standardwert ist
            (-60,60).

        edge : int
            Kantenlänge, Standardwert 100.

        thick : int
            Dicke der Streifen, Standard ist 5. Wenn thick=edge, wird ein
            homogenes Quadrat erzeugt.

        mean1 : int
            Mittlerer Grauwert der Streifen mit ungerader Zahl (1., 3., etc).
            Standardwert ist 250.

        mean2 : int
            Mittlerer Grauwert der Streifen mit gerader Zahl. Standardwert
            ist 300.

        Beschreibung:
        -----------------------------------------------------------------------
        Erzeugt ein Quadrat mit gegebener Kantenlänge um das Zentrum herum.
        Es können Streifen unterschiedlicher Flächendichte vorgegeben werden.

        Ausgabe
        -----------------------------------------------------------------------
        Das spezifizierte Quadrat wird in das image-Array der Instanz
        übertragen. Keine weitere Ausgabe.
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
        Parameter
        -----------------------------------------------------------------------
        center : array_like
            Mittelpunkt des Kreises in der Form [x,y], Standardwert ist
            (-60,-60).

        radius : int
            Radius des zu erzeugenden Kreises. Standardwert ist 50.

        mean : int
            Mittlerer Grauwert des Kreises. Standardwert ist 50.

        Beschreibung
        -----------------------------------------------------------------------
        Erzeugt Kreis um gegebenes Zentrum.

        Ausgabe
        -----------------------------------------------------------------------
        Kreis wird in self.image eingetragen, keine weitere Ausgabe.
        """
        c_xy = self.get_center(center)
        mask = np.fromfunction(lambda x,y: (c_xy[1]-x)**2 + (c_xy[0]-y)**2,
                               self.image.shape) < radius**2
        self.image[mask] = np.random.poisson(mean,self.image.shape)[mask]

    def triangle(self,top=(60,-10),height=100,mean=100):
        """
        Parameter:
        -----------------------------------------------------------------------
        top : array_like
            Ort der Dreieckspitze, in der Form [x,y]. Standardwert ist (60,-10).

        height : int
            Höhe des Dreiecks von Grundseite bis Spitze, Standardwert 100.

        mean : int
            Mittlerer Grauwert, Standardwert 100.

        Beschreibung
        -----------------------------------------------------------------------
        Erzeugt ein Dreieck unterhalb des gegebenen Punktes.

        Ausgabe
        -----------------------------------------------------------------------
        Dreieck wird in self.image eingetragen, keine weitere Ausgabe.
        """
        top = self.get_center(top)
        side = height/np.sin(60/360. * 2 * np.pi)
        mask = np.fromfunction(lambda x,y: (np.abs(top[0]-y)<=(
        np.abs(top[1]-x)/height*side/2))*((top[1]-x) <= 0)*
        ((top[1]-x) >= -height),self.image.shape)
        self.image[mask] = np.random.poisson(mean,self.image.shape)[mask]

class grayscale:

    def __init__(self,size=256):
        """
        Parameter
        -----------------------------------------------------------------------
        size : int
            Kantenlänge des quadratischen Graukeils.

        Funktionen
        -----------------------------------------------------------------------
        inverse :
            Inverse Kennlinie.

        squared :
            Quadratische Kennlinie.

        root :
            Wurzelfunktions-Kennlinie.

        binary :
            Binäre Kennlinie.

        gaussian :
            Gauß-Kennlinie.

        Beschreibung
        -----------------------------------------------------------------------
        Erstellt einen quadratischen Graukeil mit 256 Grauwerten, der durch
        verschiedene Kennlinien manipuliert werden kann.

        Ausgabe
        -----------------------------------------------------------------------
        Keine weitere Ausgabe.
        """
        self.size = size
        self.image = np.ones((size,size)) * np.arange(0,size).astype(int)

    @classmethod
    def inverse(self,image):
        """
        Beschreibung
        -----------------------------------------------------------------------
        Wendet inverse Kennlinie auf Bild an.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            invertierte Bilddaten.
        """
        return 255 - image

    @classmethod
    def squared(self,image):
        """
        Beschreibung
        -----------------------------------------------------------------------
        Wendet quadratische Kennlinie an.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Bilddaten nach quadratischer Kennlinie.
        """
        return np.round(image**2 / 255.,0)

    @classmethod
    def root(self,image):
        """
        Beschreibung
        -----------------------------------------------------------------------
        Wendet die Wurzelfunktion als Kennlinie an.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Bilddaten nach Wurzelkennlinie.
        """
        return np.round(np.sqrt(image*255),0)

    @classmethod
    def binary(self,image,lower,upper):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Die zu binarisierenden Bilddaten

        lower : int
            Untere Grenze des Grauwertebereichs, der auf 0 gesetzt werden soll.

        upper : int
            Obere Grenze des auf 1 zu setzenden Grauwertebereichs.

        Beschreibung
        -----------------------------------------------------------------------
        Binarisiert das Bild, wobei alle Werte zwischen lower und upper als 1
        gesetzt werden.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Binarisierte Bilddaten.
        """
        return (image > lower) * (image < upper)

    @classmethod
    def gaussian(self,image,sigma=85.,mu=0.):
        """
        Parameter
        -----------------------------------------------------------------------
        sigma : float
            Standardabweichung der in der Kennlinie verwendeten Gaußfunktion.
            Standardwert ist 85.

        mu : float
            Mittelwert der Gaußfunktion. Standardwert ist 0.

        Beschreibung
        -----------------------------------------------------------------------
        Gauß-Kennlinie.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Bilddaten nach Gauß-Kennlinie.
        """
        return np.round(258 - 54942/(np.sqrt(2*np.pi)*sigma) *
            np.exp(-(image - mu)**2/(2*sigma**2)),0)


class tools:

    @classmethod
    def bar(self,x,y,title="",label_x="",label_y="",
             label_data="",color="blue",fig=1):
        """
        Parameter
        -----------------------------------------------------------------------
        x,y : ndarray
            Achsen des anzufertigenden Balkendiagramms.

        fig : int
            Nummer der zu verwendenden pyplot-Figure. Standardwert ist 1.

        color : string
            Die von plt.bar zu verwendende Farbe. Standardwert ist
            "blue".

        title,label_x,label_y,label_data : string
            Der in der Grafik verwendete Titel, die Achsenbezeichnungen und
            die Graphbezeichnung.

        Beschreibung
        -----------------------------------------------------------------------
        Einfache Möglichkeit, die erzeugten Daten auszugeben.

        Ausgabe
        -----------------------------------------------------------------------
        pyplot-Figure.
        """
        plt.figure(fig)
        plt.bar(x,y,color=color,label=label_data)
        plt.title(title)
        plt.ylabel(label_y)
        plt.xlabel(label_x)
        plt.legend()

        plt.show()

    @classmethod
    def plot(self,x,y,title="",label_x="",label_y="",
             label_data="",color="blue",fig=1,lw=1):
        """
        Parameter
        -----------------------------------------------------------------------
        x,y : ndarray
            Achsen des anzufertigenden Plots

        fig : int
            Nummer der zu verwendenden pyplot-Figure. Standardwert ist 1.

        color : string
            Die von plt.plot zu verwendende Farbe. Standardwert ist
            "blue".

        title,label_x,label_y,label_data : string
            Der in der Grafik verwendete Titel, die Achsenbezeichnungen und
            die Graphbezeichnung.

        Beschreibung
        -----------------------------------------------------------------------
        Einfache Möglichkeit, die erzeugten Daten auszugeben.

        Ausgabe
        -----------------------------------------------------------------------
        pyplot-Figure.
        """
        plt.figure(fig)
        plt.plot(x,y,color=color,label=label_data,linewidth=lw)
        plt.title(title)
        plt.ylabel(label_y)
        plt.xlabel(label_x)
        plt.legend()

        plt.show()

    @classmethod
    def show(self,image,title="",color="gray",fig=1):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddaten die angezeigt werden sollen.

        fig : int
            Nummer der zu verwendenden pyplot-Figure. Standardwert ist 1.

        color : string
            Die von plt.imshow zu verwendende Farbpalette. Standardwert ist
            "gray".

        title : string
            Der in der Grafik verwendete Titel.

        Beschreibung
        -----------------------------------------------------------------------
        Einfache Möglichkeit, die erzeugten Bilder auszugeben.

        Ausgabe
        -----------------------------------------------------------------------
        pyplot-Figure.
        """
        plt.figure(fig)
        plt.imshow(image,cmap=color)
        plt.title(title)
        plt.axis("off")

        plt.show()

    @classmethod
    def show_subplot(self,images,titles,dimensions,color="gray",fig=1):
        """
        Parameter
        -----------------------------------------------------------------------
        images : list of ndarrays
            Bilddaten die angezeigt werden sollen, ein Bild pro Element.

        titles : list of strings
            Die zu den Bildern gehörigen Überschriften, gleiche Reihenfolge
            wie in 'images'

        dimensions : list or tuple
            Die in der Grafik zu verwendende Anordnung.

        fig : int
            Nummer der zu verwendenden pyplot-Figure. Standardwert ist 1.

        color : string
            Die von plt.imshow zu verwendende Farbpalette. Standardwert ist
            "gray".

        Beschreibung
        -----------------------------------------------------------------------
        Bequemlichkeitsfunktion, um die Erzeugung von Bildern mit Subplots
        einigermaßen zu automatisieren.

        Ausgabe
        -----------------------------------------------------------------------
        pyplot-Figure.
        """
        plt.figure(fig)
        for num in range(len(images)):
            plt.subplot(dimensions[0],dimensions[1],num+1)
            plt.title(titles[num])
            plt.imshow(images[num],cmap="gray")
            plt.axis("off")

    @classmethod
    def profile(self,image,y=0):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Die Bilddaten, durch die ein Profil gelegt werden soll.
        y : int
            y-Koordinate des Profils, wird in Array-Index umgerechnet.

        Beschreibung
        -----------------------------------------------------------------------
        Legt ein Profil bei der angegebenen y-Koordinate durch das gesamte
        Bild.

        Ausgabe
        -----------------------------------------------------------------------
        result : [[ndarray,ndarray],[ndarray,ndarray]]
            Liste aus zwei Elementen. Erstes Element enthält die Achsen eines
            Diagramms um die Profillage im Ursprungsbild einzuzeichnen, zweites
            Element das Profil an sich.
        """

        height,width = image.shape
        result = []

        profile_location = np.ones(width)*(height/2 - y)

        result.append([np.arange(width),profile_location])
        result.append([np.arange(width),image[-y+height/2,:]])

        return result

    @classmethod
    def histogram(self,image,normed=False,bin_count=256):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddaten, deren Histogramm erzeugt werden soll.

        exlude_zero : boolean
            Bestimmt, ob im Histogramm der 0-Wert auftaucht oder nicht.

        bin_count : int
            Anzahl der Bins im Histogramm. Standardwert ist 256.

        Beschreibung
        -----------------------------------------------------------------------
        Erzeugt ein Histogram der in den Daten vorhandenen Grauwerte. Ausgabe
        in beliebiger Figure. Per exclude_zero=True kann die Berechnung des
        Histograms auf Werte > 0 beschränkt werden, um eine detailliertere
        Darstellung des variablen Wertebereichs zu erhalten.

        Ausgabe
        -----------------------------------------------------------------------
        output : list of ndarrays
            Erstes Element enthält die Histogramm-Bins, zweites Element die
            (relativen) Häufigkeiten je nach Einstellung.
        """
        histo, bins = np.histogram(image,bins=bin_count)
        if normed:
            return [bins[:-1],histo/65536.]

        return [bins[:-1],histo]

    @classmethod
    def mean_skew(self,histogram):
        """
        Parameter
        -----------------------------------------------------------------------
        histogram : array_like
            histogram[0] muss die Bins und histogram[1] die entsprechenden
            Häufigkeiten (normiert) enthalten.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet Mittelwert und Schiefe des Grauwerthistogramms.

        Ausgabe
        -----------------------------------------------------------------------
        output : list
            Liste mit [Mittelwert,Schiefe].
        """
        bins, histo = histogram

        mean = np.sum(histo * bins)
        skew = np.sum((bins - mean)**3*histo)

        return np.round([mean,skew],3)

    @classmethod
    def mean_information(self,histogram):
        """
        Parameter
        -----------------------------------------------------------------------
        histogram : array_like
            histogram[0] muss die Bins und histogram[1] die entsprechenden
            Häufigkeiten (normiert) enthalten.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet die mittlere Information je Pixel, basierend auf dem
        übergebenen Histogram.

        Ausgabe
        -----------------------------------------------------------------------
        output : float
            Mittlere Information, gerundet auf 3 Stellen.
        """
        bins, histo = histogram
        histo = np.sum(-1*histo[histo > 0]*np.log2(histo[histo>0]))
        return np.round(histo,3)

    @classmethod
    def bit_layer(self,image,layer=0):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddaten, deren Bitebene berechnet werden soll.

        layer : int
            Welche Bitebene ausgegeben wird.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet zu gegebenem Datensatz die Bitebenen 0 bis 7.

        Ausgabe
        -----------------------------------------------------------------------
        bitdata : ndarray
            Enthält die spezifizierte Bitebene als Array.
        """
        height,width = image.shape

        bitdata = np.unpackbits(image.astype(np.uint8)).reshape(height,-1)
        return bitdata[:,7-layer::8]

    @classmethod
    def difference(self,image):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet ein zeilenweises Differenzbild mit Verschiebung 1 bei
        gegebenen Bilddaten.

        Ausgabe
        -----------------------------------------------------------------------
        diff_image : ndarray
            Das berechnete Differenzbild.
        """
        diff_image = image[:,1:] - image[:,:-1]
        return diff_image - np.min(diff_image)

    @classmethod
    def fft2(self,image):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        Beschreibung
        -----------------------------------------------------------------------
        Führt eine 2D-Fouriertransformation durch und wendet fftshift auf das
        Ergebnis an. Reine Bequemlichkeitsfunktion.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die geshiftete Fouriertransformierte von image.
        """
        return np.fft.fftshift(np.fft.fft2(image))

    @classmethod
    def ifft2(self,image):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        Beschreibung
        -----------------------------------------------------------------------
        Führt den Rückweg von tools.ifft2 aus. Zunächst also ifftshift, dann
        ifft2 angewendet.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die Rücktransformierte von image.
        """
        return np.fft.ifft2(np.fft.ifftshift(image))

    @classmethod
    def lowpass(self,image,cutoff=0.25):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        cutoff : float, < 1
            Der Bruchteil der Nyqist-Frequenz, bis zu dem noch durchgelassen
            wird. Standardwert ist 0.25.

        Beschreibung
        -----------------------------------------------------------------------
        Wendet einen Tiefpass auf die übergebenen Daten an. Cutoff ist variabel,
        wird noch mit 0.5 multipliziert um der Ausgabe von fft2 Rechnung zu
        tragen (Werte für Nyqvist-Frequenz liegen nach fftshift am Rand).

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die Tiefpass-gefilterten Daten.
        """
        height,width = image.shape

        ft_data = self.fft2(image)
        lowpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)
        lowpass = np.array(lowpass < height*(0.5*cutoff),dtype=int)
        return self.ifft2(lowpass * ft_data)

    @classmethod
    def bandpass(self,image,cutoff=[0.375,0.525]):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        cutoff : list, cutoff[0] < cutoff[1]
            Die Unter- und Obergrenze des Durchlassbereiches.

        Beschreibung
        -----------------------------------------------------------------------
        Bandpass, ansonsten identisch zu lowpass.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die Bandpass-gefilterten Bilddaten.
        """
        height,width = image.shape
        ft_data = self.fft2(image)

        bandpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)
        bandpass = np.array((bandpass > height * cutoff[0]*0.5)*
            (bandpass < height * cutoff[1]*0.5),dtype=int)

        return self.ifft2(ft_data*bandpass)

    @classmethod
    def highpass(self,image,cutoff=0.75):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        cutoff : float
            Die untere Grenze des Durchlassbereiches.

        Beschreibung
        -----------------------------------------------------------------------
        Hochpass, ansonsten identisch zu lowpass.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die Hochpass-gefilterten Bilddaten.
        """
        height,width = image.shape
        ft_data = self.fft2(image)

        highpass = np.fromfunction(lambda x,y:np.sqrt((x-width/2)**2+
            (y-height/2)**2),image.shape)
        highpass = np.array(highpass > height*cutoff*.5,dtype=int)

        return self.ifft2(ft_data*highpass)

    @classmethod
    def shear(self,image):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        Beschreibung
        -----------------------------------------------------------------------
        Führt eine festgelegte Scherung auf die übergebene Bildmatrix aus.
        Die Bildgröße ändert sich durch die Scherung.

        Ausgabe
        -----------------------------------------------------------------------
        shear : ndarray
            Das gescherte Bild. Dimensionen der Ausgabe stimmen nicht mit denen
            der Eingabe überein.

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
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddatensatz.

        Beschreibung
        -----------------------------------------------------------------------
        Erzeugt ein Array das für jeden Pixel des input-Bildes alle 9 ELemente
        der 9er-Nachbarschaft enthält.

        Vorteil: Keine Schleifen, vermutlich schnell.
        Nachteil: 9facher Speicherplatz des Ausgangsbildes benötigt.

        Ausgabe
        -----------------------------------------------------------------------
        neighbors : ndarray
            Reihenfolge: (y,x),(y,x+1),(y-1,x+1),(y-1,x),(y-1,x-1),(y,x-1),
        (y+1,x-1),(y+1,x),(y+1,x+1) für neighbors[0] bis neighbors[8].

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
    def mean_filter(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        9 Punkte Mittelwertsfilter.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Enthält für jeden Pixel im Ursprungsbild den Mittelwert der 3x3-
            Umgebung.
        """
        return np.mean(neighbors,axis=0)

    @classmethod
    def median_filter(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        9 Punkte Medianfilter.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Enthält für jeden Pixel im Ursprungsbild den Median der 3x3-
            Umgebung.
        """
        return np.median(neighbors,axis=0)

    @classmethod
    def gauss_filter(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        9 Punkte Binomialfilter.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Enthält für jeden Pixel im Ursprungsbild das Ergebnis eines 3x3
            Binomial-Filters.

        """
        gauss = 1*neighbors
        gauss[0] *= 4
        gauss[1::2] *= 2
        return 1/16. * np.sum(gauss,axis=0)

    @classmethod
    def laplace_filter_8(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        9 Punkte Laplace-Filter.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Enthält für jeden Pixel im Ursprungsbild das Ergebnis eines 3x3
            Laplace-Filters.
        """
        laplace = 1*neighbors
        laplace[0] *= 8
        laplace[1:] *= -1
        return np.sum(laplace,axis=0)

    @classmethod
    def sobel_filter(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        Sobel-Filter, also eine gewichtete 1. Ableitung im Bildraum. Der
        Filterkern ist 3x3 Pixel groß.

        Ausgabe
        -----------------------------------------------------------------------
        output : list of ndarrays
            Enthält die absoluten Grauwerte nach Anwendung des Sobel-Filters
            in x- und y-Richtung, [sobel_x,sobel_y].
        """
        sobel_x = neighbors[[1,2,4,5,6,8]]
        sobel_x[[0,3]] *= 2
        sobel_x[[2,3,4]] *= -1

        sobel_y = neighbors[[2,3,4,6,7,8]]
        sobel_y[[1,4]] *= 2
        sobel_y[[0,1,2]] *= -1

        return np.abs([np.sum(sobel_x,axis=0),np.sum(sobel_y,axis=0)])

    @classmethod
    def roberts_filter(self,neighbors):
        """
        Parameter
        -----------------------------------------------------------------------
        neighbors : ndarray
            Output aus tools.neighbors_3x3() für das zu filternde Bild.

        Beschreibung
        -----------------------------------------------------------------------
        Roberts-Filter, es werden die Kerne [[1,0],[0,-1]] und [[0,1],[-1,0]]
        auf die Bilddaten angewandt.

        Ausgabe
        -----------------------------------------------------------------------
        output : list of ndarrays
            Enthält die absoluten Grauwerte nach Anwendung des Roberts-Filters.
            in x- und y-Richtung, [roberts_x,roberts_y].
        """
        roberts_x = neighbors[0] - neighbors[8]
        roberts_y = neighbors[1] - neighbors[7]

        return np.abs([roberts_x,roberts_y])

#    @classmethod
#    def threshold(self,image):
#


    @classmethod
    def image_moment(self,image,p,q):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bilddaten, deren Moment berechnet werden soll.

        p,q : int
            Es wird das Moment m_pq berechnet.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet das Moment m_pq zu gegebenem Bild. Hierbei wird stets der
        jeweilige Grauwert mit in die Berechnung einbezogen, für
        Massenschwerpunkte ist also vorher eine Binarisierung vorzunehmen.

        Ausgabe
        -----------------------------------------------------------------------
        output : float
            Das pq-te Moment des Eingabebildes.
        """
        height,width = image.shape
        yx = np.mgrid[0:height,0:width]

        return np.sum(image*yx[0]**q*yx[1]**p).astype(float)

    @classmethod
    def transition_matrix(self,image,delta):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bild, dessen Grauwerte-Übergangsmatrix berechnet werden soll.

        delta : array_like
            Verschiebungsvektor, entlang dem die Übergänge betrachtet werden.

        Beschreibung
        -----------------------------------------------------------------------
        Berechnet die Grauwerte-Übergangsmatrix für 'image' entlang 'delta'.

        Achtung: Der Einfachheit halber wird zur Berechnung das Bild mit einem
        Ein-Pixel-Rand mit Wert -999 versehen. Sollte dieser Wert im
        Ausgangsbild vorkommen, ist mit verfälschten Ergebnissen zu rechnen.

        Ausgabe
        -----------------------------------------------------------------------
        output : ndarray
            Die Grauwerte-Übergangsmatrix.
        """
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
                transition_matrix[m,n] = np.sum(
                working_image[index[0]+delta[1],index[1]+delta[0]] == bins[n])

        return transition_matrix

    @classmethod
    def hough(self,image,angle_count=180):
        """
        Parameter
        -----------------------------------------------------------------------
        image : ndarray
            Bild, dessen Hough-Transformation berechnet werden soll.

        angle_count : array_like
            Anzahl an Werten zwischen 0 und pi, die geprüft werden.

        Beschreibung
        -----------------------------------------------------------------------
        Führt eine Hough-Transformation zur Kantenerkennung durch. Die Kanten
        werden zunächst mit dem Roberts-Filter extrahiert, anschließend wird
        die Hough-Transformation ausgeführt. Aus dem Ergebnis lassen sich die
        Winkel aller vorhandenen Kanten ablesen.

        Ausgabe
        -----------------------------------------------------------------------
        output : tupel of ndarrays
            Enthält als erstes Element das Diagramm der Hough-Transformation als
            nicht-normalisiertes Graustufenbild, als zweites Element die y-Achse
            eines Diagramms dessen Minima die Winkelkoordinate der Häufungs-
            punkte darstellen.
        """
        edges = np.max(self.roberts_filter(self.neighbors_3x3(image)),axis=0)\
            > 50

        height,width = image.shape

        y,x = np.where(edges > 0)
        alpha = np.linspace(0,np.pi,angle_count)

        d = np.array([x*np.cos(angle) +
            y*np.sin(angle) for angle in alpha]).astype(int)
        #entsprechend der Hesseschen Normalenform.

        hough = np.zeros((np.max(d)-np.min(d)+1,angle_count))

        for angle in range(angle_count):
            index,count = np.unique(d[angle],return_counts=True)
            hough[np.max(d)-index,angle] += count

        return hough,np.sum(hough > 0,axis=0)

if __name__ == "__main__":
    hline = "--------------------------------------------------"
    plt.close("all")
    pic = scintigram()
    img = pic.image
    neighbors = tools.neighbors_3x3(img)
###############################################################################
    #Aufgabe 1.1
###############################################################################
    tools.show(img,"Aufgabe 1.1",fig=1)

###############################################################################
    #Aufgabe 2.1
###############################################################################
    profile_1 = tools.profile(img,60)
    profile_2 = tools.profile(img,-60)
    tools.show(img,"Aufgabe 2.1",fig=2)
    plt.plot(profile_1[0][0],profile_1[0][1],color="red")
    plt.plot(profile_2[0][0],profile_2[0][1],color="blue")

    tools.plot(profile_1[1][0],profile_1[1][1],
               "Aufgabe 2.1","Pixel","Grauwert","y = 60","red",3)
    tools.plot(profile_2[1][0],profile_2[1][1],
               "Aufgabe 2.1","Pixel","Grauwert","y = -60","blue",3)

###############################################################################
    #Aufgabe 2.2
###############################################################################
    hist = tools.histogram(img,True)

    tools.bar(hist[0],hist[1],"Aufgabe 2.2","Grauwert",
              "h(f)","Bild aus 1.1",fig=4)

###############################################################################
    #Aufgabe 2.3
###############################################################################
    mean,skew = tools.mean_skew(hist)
    print("\n\n{0}\nAufgabe 2.3\n{0}\nMittelwert \t\tSchiefe\n"
    "{1} \t\t{2}"\
    .format(hline,mean,skew))

###############################################################################
    #Aufgabe 2.4
###############################################################################
    print("\n{0}\nAufgabe 2.4\n{0}\nMittlerer Informationsgehalt: {1}"\
    .format(hline,tools.mean_information(hist)))

###############################################################################
    #Aufgabe 2.5
###############################################################################
    layers = [tools.bit_layer(img,num) for num in range(8)]
    layers.insert(0,img)
    means = []
    titles = ["Original"]
    min_value = np.min(layers[num+1])

    for num in range(8):
        bithist = tools.histogram(layers[num+1],normed=True,bin_count=2)
        titles.append("Bitebene {0}".format(num))
        means.append(tools.mean_information([bithist[0],bithist[1]/2.]))
    means = np.round(means,3)

    tools.show_subplot(layers,titles,(3,3),fig=6)

    print("\n{0}\nAufgabe 2.5\n{0}\nBitebene \t\t0 \t1 \t2 \t3 \t4 \t5 \t6 \t7\n"
    "Information \t{1} \t{2} \t{3} \t{4} \t{5} \t{6} \t{7} \t{8}".format(
    hline,*means))

###############################################################################
    #Aufgabe 2.6
###############################################################################
    diff_img = tools.difference(img)
    tools.show(diff_img,"Aufgabe 2.6, Differenzbild",fig=7)

    diff_hist = tools.histogram(diff_img,True)
    tools.bar(diff_hist[0],diff_hist[1],"Aufgabe 2.6, Histogramm","Grauwert",
              "h(f)","Differenzbild",fig=8)
    print("\n{0}\nAufgabe 2.6\n{0}\nMittlerer Informationsgehalt: {1}"
    .format(hline,tools.mean_information(diff_hist)))

###############################################################################
    #Aufgabe 2.7
###############################################################################
    ft_img = tools.fft2(img)
    tools.show(np.abs(ft_img[100:156,100:156]),
               "Aufgabe 2.7, |Fouriertransformierte|",fig=10)
    tools.show(np.abs(ft_img)[100:156,100:156]**2,
              "Aufgabe 2.7, Leistungsspektrum",fig=11)

###############################################################################
    #Aufgabe 2.8
###############################################################################
    rot_img = sip.rotate(img,30)
    tools.show(rot_img,"Aufgabe 2.8, rotiertes Original",fig=12)
    tools.show(np.abs(tools.fft2(rot_img))[150:200,150:200],
               "Aufgabe 2.8,|Fouriertransformierte|",fig=13)

###############################################################################
   #Aufgabe 2.9
###############################################################################
    tools.show(np.abs(tools.lowpass(img,0.25)),"Aufgabe 2.9",fig=14)

###############################################################################
    #Aufgabe 2.10
###############################################################################
    tools.show(np.abs(tools.bandpass(img,[0.375,0.625])),"Aufgabe 2.10",fig=15)

###############################################################################
    #Aufgabe 2.11
###############################################################################
    tools.show(np.abs(tools.highpass(img,0.75)),"Aufgabe 2.11",fig=16)

###############################################################################
    #Aufgabe 3.1
###############################################################################
    g = grayscale()
    grays = [g.image]
    grays.append(grayscale.inverse(g.image))
    grays.append(grayscale.squared(g.image))
    grays.append(grayscale.root(g.image))
    grays.append(grayscale.binary(g.image,64,128))
    grays.append(grayscale.gaussian(g.image))
    titles = ["Linear","Invers","Quadratisch","Wurzel","Binarisiert","Gauss"]

    tools.show_subplot(grays,titles,(3,2),fig=17)

###############################################################################
    #Aufgabe 3.2
###############################################################################
    tools.show(tools.shear(sip.rotate(img,90)),"Aufgabe 3.2",fig=18)

###############################################################################
    #Aufgabe 3.3
###############################################################################

    filters = [img,tools.mean_filter(neighbors)]
    filters.append(tools.median_filter(neighbors))
    filters.append(tools.gauss_filter(neighbors))

    profiles = [tools.profile(filtered,60) for filtered in filters]

    titles = ["Original","Mittelwert","Median","Binomial"]

    tools.show_subplot(filters,titles,(2,2),fig=19)

    for num in range(4):
        plt.figure(19)
        plt.subplot(2,2,num+1)
        plt.plot(profiles[num][0][0],profiles[num][0][1])
        plt.axis("off")
        plt.figure(20)
        plt.subplot(2,2,num+1)
        plt.title(titles[num])
        plt.plot(profiles[num][1][0],profiles[num][1][1])

###############################################################################
    #Aufgabe 3.4
###############################################################################
    images = [img,np.sum(tools.sobel_filter(neighbors),axis=0),
              np.sum(tools.roberts_filter(neighbors),axis=0)]
    titles = ["Original","Sobel-Filter","Roberts-Filter"]
    tools.show_subplot(images,titles,(1,3),fig=21)

###############################################################################
    #Aufgabe 3.5
###############################################################################
    tools.show(tools.laplace_filter_8(neighbors),"Aufgabe 3.5, Laplace-Filter",
               fig=22)

###############################################################################
   #Aufgabe 3.6
###############################################################################
    plt.figure(23)
    plt.subplot(1,2,1)
    tools.bar(hist[0][1:],hist[1][1:],"Histogramm","Grauwert","h(f)",
              "Originalbild", fig=23)
    tools.plot(55*np.ones(50),np.linspace(0,max(hist[1][1:]),50),
               label_data="untere Grenze",color="green",fig=23,lw=2)
    tools.plot(95*np.ones(50),np.linspace(0,max(hist[1][1:]),50),
           label_data="obere Grenze",color="red",fig=23,lw=2)
    plt.subplot(122)
    tools.show(grayscale.binary(img,55,95),"Ergebnis nach Schwellwert",fig=23)

###############################################################################
    #Aufgabe 3.7
###############################################################################
    hough,corridor = tools.hough(img[128:,128:],180)
    tools.show(hough,"Aufgabe 3.7, Ergebnis der Hough-Transformation",fig=24)
    tools.plot(range(180),corridor,"Aufgabe 3.7, Korridorbreite","Winkel",
               "","Korridorbreite",fig=25)

###############################################################################
    #Aufgabe 3.8
###############################################################################
    BC = img[:,5:131]
    g00,g10,g01 =\
        tools.image_moment((BC > 0),0,0),\
        tools.image_moment((BC > 0),1,0),\
        tools.image_moment((BC > 0),0,1)

    m00,m10,m01 =\
        tools.image_moment(BC,0,0),\
        tools.image_moment(BC,1,0),\
        tools.image_moment(BC,0,1)

    g_x, g_y = g10/g00,g01/g00
    m_x, m_y = m10/m00,m01/m00
    #entsprechend der Formel aus dem Skript werden die per Funktion berechneten
    #Momente in x- und y-Koordinaten umgesetzt.

    tools.show(BC,"",fig=26)
    plt.plot(g_x,g_y,"ro",label="Geometrischer Schwerpunkt",linewidth=2)
    plt.plot(m_x,m_y, "bo",label="Massenschwerpunkt",linewidth=2)
    plt.legend()

###############################################################################
    #Aufgabe 3.9
###############################################################################
    B = tools.median_filter(neighbors)[17:117,17:117]
    matrix10 = tools.transition_matrix(B,(1,0))
    matrix01 = tools.transition_matrix(B,(0,1))
    tools.show_subplot([B,matrix10,matrix01],
                       ["Original","d=(1,0)","d=(0,1)"],(1,3),fig=27)