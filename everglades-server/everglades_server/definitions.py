import numpy as np

class EvgMap:
    def __init__(self, name):
        self.name = name
        self.nodes = []

class EvgMapNode:
    def __init__(self, **kwargs):
        self.ID = kwargs.get('ID',-1)
        self.radius = kwargs.get('radius',1)
        self.resource = kwargs.get('resource',[])
        self.defense = kwargs.get('defense',1)
        self.controlPoints = kwargs.get('points',100)
        self.teamStart = kwargs.get('teamStart',-1)
        self.controlledBy = self.teamStart
        self.controlState = 0
        self.connections = []
        self.connection_idxs = []
        self.groups = {}
        self.groups[0] = []
        self.groups[1] = []

class EvgNodeConnection:
    def __init__(self, **kwargs):
        self.destID = kwargs.get('destID',-1)
        self.distance = kwargs.get('distance',1)

class EvgPlayer:
    def __init__(self, playerNum):
        self.playerNum = playerNum
        self.ready = True
        self.groups = []

class EvgGroup:
    def __init__(self, **kwargs):
        self.groupID = kwargs.get('groupID',-1)
        self.mapGroupID = kwargs.get('mapGroupID',-1)
        self.mapUnitID = kwargs.get('mapUnitID',-1)
        self.location = kwargs.get('location',-1)
        self.ready = False
        self.moving = False
        self.destroyed = False
        self.distance_remaining = 0
        self.travel_destination = -1
        self.units = []
        self.pathIndex = 0

class EvgUnitDefinition:
    def __init__(self, **kwargs):
        self.unitType = kwargs.get('name',None)
        self.health = kwargs.get('health',0)
        self.damage = kwargs.get('damage',0)
        self.speed = kwargs.get('speed',0)
        self.control = kwargs.get('control',0)
        self.cost = kwargs.get('cost',0)

class EvgUnit:
    def __init__(self, **kwargs):
        self.unitType = kwargs.get('unitType',-1)
        self.count = kwargs.get('count',0)
        self.unitHealth = np.ones(self.count) * 100.
        # Set null values for type, health, damage, speed, control, and cost
        self.definition = EvgUnitDefinition()
