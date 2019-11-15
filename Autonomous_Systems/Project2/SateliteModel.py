#satelite model

class SateliteModel:


    def __init__(self, satelite_locations, sigma = 1):
        self.sigma = sigma #noise
        self.satelites_locations = satelite_locations #NX3 matrix whre N number satelites, x,y,z positions

    def orbitSatelites(self): #moves the satelite positions

    def getPosition(self,reference): #returns the estimated position of the reference

