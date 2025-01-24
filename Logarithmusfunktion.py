%matplotlib inline 
# ↑ zur Verhinderung von evtl Problemen bei mehrfachem Ausführen
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('_mpl-gallery') #fürs Diagramm
fig, ax = plt.subplots() #fürs Diagramm

# Werte vom Abkühlen des Arduinos (v3)
# Zeit t
#x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147])
# Temperatur T
#y = np.array([23.73,22.95,22.42,21.89,21.45,21.1,20.76,20.25,19.77,19.34,18.96,18.55,18.24,17.97,17.66,17.36,17.1,16.87,16.78,16.48,16.26,16.1,15.97,15.76,15.53,15.37,15.33,15.3,15.18,14.95,14.58,14.26,14.05,13.98,13.78,13.57,13.39,13.29,13.08,12.82,12.62,12.46,12.33,12.32,12.26,12.2,12.02,11.92,11.9,11.82,11.71,11.63,11.57,11.41,11.28,11.15,11.02,10.84,10.71,10.54,10.42,10.35,10.24,10.11,10.02,9.94,9.79,9.69,9.6,9.52,9.41,9.31,9.18,9.03,8.92,8.84,8.71,8.5,8.48,8.37,8.27,8.18,8.05,7.93,7.82,7.72,7.61,7.51,7.4,7.29,7.17,7.09,7,6.89,6.82,6.7,6.51,6.47,6.46,6.36,6.26,6.16,6.05,5.97,5.87,5.81,5.71,5.61,5.5,5.42,5.34,5.24,5.1,4.97,4.86,4.71,4.62,4.59,4.52,4.47,4.39,4.35,4.28,4.21,4.18,4.09,4.04,3.97,3.91,3.83,3.76,3.75,3.61,3.56,3.48,3.4,3.35,3.26,3.26,3.22,3.15,3.06,2.97,2.93,2.9,2.86,2.79])


#Werte vom Abkühlen des Arduinos (fehlerfreier) (v4)
# Zeit t
x = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147])
# Temperatur T
y = np.array([16.26,16.1,15.97,15.76,15.53,15.37,15.33,15.3,15.18,14.95,14.58,14.26,14.05,13.98,13.78,13.57,13.39,13.29,13.08,12.82,12.62,12.46,12.33,12.32,12.26,12.2,12.02,11.92,11.9,11.82,11.71,11.63,11.57,11.41,11.28,11.15,11.02,10.84,10.71,10.54,10.42,10.35,10.24,10.11,10.02,9.94,9.79,9.69,9.6,9.52,9.41,9.31,9.18,9.03,8.92,8.84,8.71,8.5,8.48,8.37,8.27,8.18,8.05,7.93,7.82,7.72,7.61,7.51,7.4,7.29,7.17,7.09,7,6.89,6.82,6.7,6.51,6.47,6.46,6.36,6.26,6.16,6.05,5.97,5.87,5.81,5.71,5.61,5.5,5.42,5.34,5.24,5.1,4.97,4.86,4.71,4.62,4.59,4.52,4.47,4.39,4.35,4.28,4.21,4.18,4.09,4.04,3.97,3.91,3.83,3.76,3.75,3.61,3.56,3.48,3.4,3.35,3.26,3.26,3.22,3.15,3.06,2.97,2.93,2.9,2.86,2.79])


y = np.log(y)
averagex = np.mean(x)
averagey = np.mean(y)
maxerlabweichung = 50 # Maximal zugelassene Abweichung in %
a = 1 #Exponent für x für print
b = 1 #Exponent für y für print
c = 1 #counter
d = 1 #für while

#Funktion zur Berechnung von Durchschnitt, Min, Max und Abweichung
def min_max_abw():
    # Berechnung von Durchschnitt, Maximal- und Minimalwert
    average = np.mean(xy)
    maxxy = np.max(xy)
    meanmaxxy = ( maxxy + average ) / 2 #zur Verringerung des Einflusses von Messfehlern
    minxy = np.min(xy)
    meanminxy = ( minxy + average ) / 2 #zur Verringerung des Einflusses von Messfehlern

    # Maximalen Abweichung vom Durchschnitt
    abweichungnumb = np.abs([meanmaxxy - average, meanminxy - average])
    maxabweichungnumb = np.max(abweichungnumb)

    #Abweichung in %
    abweichung = (maxabweichungnumb / average) * 100

    return average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy

# im Falle eines antiproportinalen Zusammenhangs
def antiproportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy):
    # Überprüfung der Abweichung & Ergebnis ausgeben
    if abweichung < maxerlabweichung:
        print("\n - x entspricht T (Temperatur); y entspricht t (Zeit) - ")
        print("Durchschnitt von (x ↑",a,"* y ↑",b,"):", average)
        print("Maximale Abweichung vom Durchschnitt (%) (fehlerbereinigt):", abweichung, "%")
        print("Maximal zugelassene Abweichung:", maxerlabweichung, "%")
        print("\nErgebnis:")
        
        print("(x ↑",a,"* y ↑",b,") = konstant (antiproportional)")
        print("(Die Abweichung vom Durchschnitt überschreitet die maximal zugelassene Abweichung nicht)")
        sys.exit()
    else:
        print("(x ↑",a,"* y ↑",b,") ist nicht konstant (nicht antiproportional)")
        print("(Die Abweichung vom Durchschnitt überschreitet die maximal zugelassene Abweichung)")
        

# im Falle eines proportionalen Zusammenhangs
def proportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy):
    # Überprüfung der Abweichung & Ergebnis ausgeben
    if abweichung < maxerlabweichung:
        print("\n - x entspricht T (Temperatur); y entspricht t (Zeit) - ")
        print("Durchschnitt von (x ↑",a,"/ y ↑",b,"):", average)
        print("Maximale Abweichung vom Durchschnitt (%) (fehlerbereinigt):", abweichung, "%")
        print("Maximal zugelassene Abweichung:", maxerlabweichung, "%")
        print("\nErgebnis:")
        
        print("(x ↑",a,"/ y ↑",b,") = konstant (proportional)")
        print("(Die Abweichung vom Durchschnitt überschreitet die maximal zugelassene Abweichung nicht)")
        sys.exit()
    else:
        print("(x ↑",a,"/ y ↑",b,") ist nicht konstant (nicht proportional)")
        print("(Die Abweichung vom Durchschnitt überschreitet die maximal zugelassene Abweichung)")

def diagramm():
    ax.set_title('log(T)(t)-Diagramm')
    ax.set_xlabel('Zeit t (in s)')
    ax.set_ylabel('log der Temperatur T (in °C)')        
    ax.plot(x, y, linewidth=2.0)
    plt.show()

try:
    diagramm()
    #Antiproportionaler Zusammenhang
    if (x[1] > x[-1] and y[1] < y[-1]) or (x[1] < x[-1] and y[1] > y[-1]):
        print("\nAntiproportionaler Zusammenhang")
        xy = x * y
        average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
        antiproportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)

        #Was soll passieren, wenn xy nicht konstantt ist (nicht antiproportional)?
        while d == 1: 
            if (xy[1] > xy[-1] and x[1] < x[-1]) or (xy[1] < xy[-1] and x[1] > x[-1]): #Antiproportional
                c = c + 1
                print("Zusammenhang", c )
                a = a + 1 #Exponent von x erhöhen
                xy = xy * x
                average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
                antiproportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)
        
            else: #Proportional
                c = c + 1
                print("Zusammenhang", c )
                b = b + 1 #Exponent von y erhöhen
                xy = xy / y
                average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
                antiproportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)

    #Proportionaler Zusammenhang
    else:
        print("\nProportionaler Zusammenhang")
        xy = x / y
        average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
        proportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)

        #Was soll passieren, wenn x/y nicht konstantt ist (nicht proportional)?
        while d == 1:
            if (xy[1] > xy[-1] and x[1] > x[-1]) or (xy[1] < xy[-1] and x[1] < x[-1]): #Proportional
                c = c + 1
                print("Zusammenhang", c )
                b = b + 1 #Exponent von y erhöhen
                xy = xy / y
                average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
                proportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)

            else: #Antiproportional
                c = c + 1
                print("Zusammenhang", c )
                a = a + 1 #Exponent von x erhöhen
                xy = xy * x
                average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy= min_max_abw()
                proportional(average, maxxy, minxy, maxabweichungnumb, abweichung, meanmaxxy, meanminxy)


except SystemExit: #Dient dem Verhindern einer Errormessage bei Verwendung von sys.exit() Fehlerbehandlung:
    pass # Keine Fehlermeldung anzeigen, wenn das Programm gestoppt wurde
