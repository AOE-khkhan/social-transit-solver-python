"""Python implementation of the CapCon transit assignment model.

This was made by converting the original Pascal source code from the paper
"F. Kurauchi, M.G.H. Bell, and J.-D. Schmoecker. Capacity constrained transit
assignment with common lines. Journal of Mathematical Modeling and Algorithms,
2:309-327, 2003", which was provided by Dr. J.-D. Schmoecker.

A few modifications have been made to better suit the needs of the main driving
algorithm, including its input and output methods, its initialization process,
and its stopping criteria.

Like the original program this reads transit network data from a local input
file, although the file format has been changed to TXT. Unlike the original
program this has been implemented as a module meant to be called by another
program rather than being run directly through a GUI. This is to make it easier
to incorporate into our main driver as a subroutine.

The only public element of this module is the CapCon class, which stores all
variables and methods necessary for running the CapCon algorithm. It has
methods to replicate the functionality of the Pascal program's GUI, including
the following:
    DataLoad() -- Reads in data from the local input folder.
    TransitUpdate() -- Updates transit data given a Freq and Capacity vector.
    Calculation() -- Evaluates the CapCon model based on the current input
        data, and outputs results to local output folder.
    CalculateFlows() -- Same as Calculation(), but returns the flow vector.
"""

import numpy as np

#==============================================================================
class _TNode:
    """A class for node objects.

    Used to associate a variety of attributes with each node, including both
    constant attributes determined by the transit network as well as values
    calculated during the iterations of the hyperpath search algorithm. All
    take default values appropriate for their data type (0 for integer, 0.0 for
    real, [] for list).

    The constructor accepts optional keyword arguments to set the attributes
    specified in the Nodedata input file. They are:
        ID -- Node ID, defaults to 0.
        Type -- Node type (0-5), defaults to 0.
        Line -- Node line ID (if applicable), defaults to 0.
        Platform -- Node platform ID (if applicable), defaults to 0.
    """

    #--------------------------------------------------------------------------
    def __init__(self, ID=0, Type=0, Line=0, Platform=0):
        """Node object constructor. Sets default values of attributes."""
        self.FID = ID # Node ID
        self.FType = Type # Node type
            # 0:Origin
            # 1:Destination
            # 2:Stop
            # 3:Boarding
            # 4:Alighting
            # 5:Failure
        self.FCost = 0.0 # Cost for node
        self.FOutArcs = [] # Set of arcs leading out of the node
        self.FInArcs = [] # Set of arcs leading into the node
        self.FFail = 0.0 # Fail-to-board probability
        self.FLine = Line # Line number (if applicable)
        self.FPlatform = Platform # Platform number (if applicable)
        self.FActiveOutArcs = [] # For use in hyperpath search

#==============================================================================
class _TArc:
    """A class for arc objects.

    Used to associate a variety of attributes with each arc, including both
    constant attributes determined by the transit network as well as values
    calculated during the iterations of the hyperpath search algorithm. All
    take default values appropriate for their data type (0 for integer, 0.0 for
    real, [] for list, False for boolean).

    The constructor accepts optional keyword arguments to set the attributes
    specified in the Arcdata input file. They are:
        ID -- Arc ID, defaults to 0.
        Type -- Arc type (0-6), defaults to 0.
        Line -- Arc line ID (if applicable), defaults to 0.
        Out -- Node ID of the arc's origin (tail), defaults to 0.
        In -- Node ID of the arc's destination (head), defaults to 0.
        Time -- Arc travel time, defaults to 0.0.
    """

    #--------------------------------------------------------------------------
    def __init__(self, ID=0, Type=0, Line=0, Out=0, In=0, Time=0.0):
        """Arc object constructor. Sets default values of attributes."""
        self.FID = ID # Arc ID
        self.FType = Type # Arc type
            # 0:Line
            # 1:Boarding Demand
            # 2:Alighting
            # 3:Stopping
            # 4:Walking
            # 5:Boarding
            # 6:Failure
        self.FTrTime = Time # Travel time
        self.FOut = Out # Node ID of arc's origin (tail)
        self.FIn = In # Node ID of arc's destination (head)
        self.FLine = Line # Line number (if applicable)
        self.FPlatform = 0 # Platform number (if applicable)
        self.FFreq = 0.0 # Service frequency
        self.FStrength = 0.0 # Link strength
        self.FCost = 0.0 # Total cost
        self.FChecked = False # Set to True if included in a hyperpath

#==============================================================================
class _TTransit:
    """A class for transit line objects.

    Used to associate a variety of attributes with each line, all of which are
    constant for the purposes of the assignment model. All take default values
    appropriate for their data type (0 for integer, 0.0 for real).

    The constructor accepts optional keyword arguments to set the attributes
    specified in the Transitdata input file. They are:
        ID -- Line ID, defaults to 0.
        Freq -- Line frequency, defaults to 0.0.
        Cap -- Line capacity, defaults to 0.0.
    """

    #--------------------------------------------------------------------------
    def __init__(self, ID=0, Freq=0.0, Cap=0.0):
        """Line object constructor. Sets default values of attributes."""
        self.FID = ID # Transit ID
        self.FFreq = Freq # Frequency
        self.FCapacity = Cap # Capacity

    #--------------------------------------------------------------------------
    def update(self, Freq, Cap):
        """Updates the line's frequency and capacity."""

        self.FFreq = Freq
        self.FCapacity = Cap

#==============================================================================
class CapCon:
    """The main public class for the CapCon module.

    This is the only public element of the CapCon module. It is meant as
    something of a replacement for the GUI window from the Pascal program. It
    contains all variables and methods necessary for running the CapCon
    algorithm which are not necessary to access outside of this evaluation.

    The constructor accepts the following keyword arguments:
        root -- A string specifying the directory containing the input and
            output folders. Defaults to the empty string, in which case the
            program will look for them in the local directory.
    """

    #--------------------------------------------------------------------------
    def __init__(self, root="", IteraMax=1000, IteraMin=100,
                 RiskParameter=10.0, ErrorThreshold=0.001):
        """CapCon object constructor. Initializes local variables.

        Accepts the following optional keyword arguments:
            root -- Root directory of network data files. Defaults to current
                working directory.
            IteraMax -- Maximum number of iterations of the overall method of
                successive averages algorithm. Defaults to 1000.
            IteraMin -- Minimum number of iterations. Defaults to 100.
            RiskParameter -- User risk-averseness parameter (called theta in
                the paper). Defaults to 10.0.
            ErrorThreshold -- Threshold for early termination of MSA algorithm.
                If the difference between consecutive solutions falls below
                this threshold, and we have already exceeded the minimum
                number of iterations, then we immediately end the main loop and
                output the solution. Defaults to 0.001.
        """

        self.root = root
        self.IteraMax = IteraMax
        self.IteraMin = IteraMin
        self.RiskParameter = RiskParameter
        self.ErrorThreshold = ErrorThreshold

        self.Nodes = [] # List of Node objects
        self.Arcs = [] # List of Arc objects
        self.Transits = [] # List of Transit objects
        self.OD = [] # List of OD traffic demands
        self.SNodes = [] # List of stopping nodes
        self.ENodes = [] # List of failure nodes
        self.DNodes = [] # List of destination nodes
        self.NodeNum = 0 # Number of nodes
        self.ArcNum = 0 # Number of arcs
        self.OriginNum = 0 # Number of origins
        self.Origins = [] # List of origin node IDs
        self.Node2Origin = []
        self.DestNum = 0 # Number of destinations
        self.Dests = [] # List of destination node IDs
        self.Node2Dest = [] # Position of node in the Dests list
        self.TransProb = np.array([]) # List of transition probabilities
        self.DArcVol = np.array([]) # Arc volume (destination-specific)
        self.DArcVol2 = np.array([])
        self.ODArcVol = np.array([]) # Arc volume (OD pair-specific)
        self.ArcVolOld = np.array([]) # Arc volume
        self.ArcVolNew = np.array([]) # " "
        self.OriginArcVol = np.array([]) # Arc volume (origin-specific)
        self.OriginArcVol2 = np.array([])
        self.DNodeVol = np.array([]) # Node volume (destination-specific)
        self.ODNodeVol = np.array([]) # Node volume (OD pair-specific)
        self.WaitArc = [] # Arc ID of wait arc on line i of platform j
        self.BoardArc = [] # Arc ID of board arc on line i of platform j
        self.Infinity = 1.0e10 # A large, finite value to treat as infinity

    #--------------------------------------------------------------------------
    def DataLoad(self):
        """Loads network data from the local input folder.

        This is equivalent to the TForm1.BLoadDataClick method from the CapCon
        Pascal program. In the Pascal implementation this was activated upon
        clicking one of the GUI buttons. Here we are making it a standalone
        method that can be activated at any time.
        """

        # Read in node data
        self.NodeNum = -1
        self.Nodes.clear()
        with open(self.root+"Nodedata.txt", 'r') as f:
            for line in f:
                self.NodeNum += 1
                if self.NodeNum > 0:
                    dum = line.split()
                    self.Nodes.append(_TNode(ID=int(dum[0]), Type=int(dum[1]),
                                       Line=int(dum[2]), Platform=int(dum[3])))

        # Read in arc data
        self.ArcNum = -1
        self.Arcs.clear()
        with open(self.root+"Arcdata.txt", 'r') as f:
            for line in f:
                self.ArcNum += 1
                if self.ArcNum > 0:
                    dum = line.split()
                    self.Arcs.append(_TArc(ID=int(dum[0]), Type=int(dum[1]),
                                     Line=int(dum[2]), Out=int(dum[3]),
                                     In=int(dum[4]), Time=float(dum[5])))

        # Read in transit data
        j = -1
        self.Transits.clear()
        with open(self.root+"Transitdata.txt", 'r') as f:
            for line in f:
                j += 1
                if j > 0:
                    dum = line.split()
                    self.Transits.append(_TTransit(ID=int(dum[0]),
                                    Freq=float(dum[1]),
                                    Cap=float(dum[2])))

        # Create incoming/outgoing arc sets for all nodes
        for a in self.Arcs:
            self.Nodes[a.FOut].FOutArcs.append(a)
            self.Nodes[a.FIn].FInArcs.append(a)

        # Create sets for each node type
        self.SNodes.clear()
        self.ENodes.clear()
        self.DNodes.clear()
        self.Origins = [0 for i in range(self.NodeNum)]
        self.Node2Origin = [0 for i in range(self.NodeNum)]
        self.Dests = [0 for i in range(self.NodeNum)]
        self.Node2Dest = [0 for i in range(self.NodeNum)]
        self.OriginNum = 0
        self.DestNum = 0
        for i in range(self.NodeNum):
            self.Node2Origin[i] = -1
            self.Node2Dest[i] = -1
            if self.Nodes[i].FType == 0:
                # Case 0: Origin
                self.Origins[self.OriginNum] = i
                self.Node2Origin[i] = self.OriginNum
                self.OriginNum += 1
            elif self.Nodes[i].FType == 1:
                # Case 1: Destination
                self.Dests[self.DestNum] = i
                self.Node2Dest[i] = self.DestNum
                self.DestNum += 1
                self.DNodes.append(self.Nodes[i])
            elif self.Nodes[i].FType == 2:
                # Case 2: Stopping
                self.SNodes.append(self.Nodes[i])
            elif self.Nodes[i].FType == 5:
                # Case 5: Failure
                self.ENodes.append(self.Nodes[i])
        self.Origins = self.Origins[0:self.OriginNum]
        self.Dests = self.Dests[0:self.DestNum]

        # Read in OD matrix data
        self.OD = [[0 for i in range(self.DestNum)]
            for j in range(self.OriginNum)]
        j = -1
        with open(self.root+"ODdata.txt", 'r') as f:
            for line in f:
                j += 1
                if j > 0:
                    dum = line.split()
                    self.OD[self.Node2Origin[int(dum[1])]
                        ][self.Node2Dest[int(dum[2])]] = int(dum[3])

        # Initialize node objects
        for n in self.Nodes:
            n.FCost = 0.0
            n.FFail = 0.0

        # Initialize arc objects
        for a in self.Arcs:
            a.FStrength = 1.0
            a.FFreq = 0.0
            if a.FType == 1:
                a.FFreq = self.Transits[a.FLine].FFreq
                a.FStrength = 1/a.FFreq

        # Initialize network arrays
        self.DNodeVol = np.zeros((self.NodeNum-self.OriginNum-1, self.DestNum),
                                 dtype=float)
        self.ODNodeVol = np.zeros((self.NodeNum-self.OriginNum-1,
                                   self.OriginNum, self.DestNum), dtype=float)
        self.DArcVol = np.zeros((self.ArcNum, self.DestNum), dtype=float)
        self.DArcVol2 = np.zeros((self.ArcNum, self.DestNum), dtype=float)
        self.ODArcVol = np.zeros((self.ArcNum, self.OriginNum, self.DestNum),
                                 dtype=float)
        self.ArcVolOld = np.zeros(self.ArcNum, dtype=float)
        self.ArcVolNew = np.zeros(self.ArcNum, dtype=float)
        self.OriginArcVol = np.zeros((self.ArcNum, self.OriginNum),
                                     dtype=float)
        self.OriginArcVol2 = np.zeros((self.ArcNum, self.OriginNum),
                                      dtype=float)
        self.TransProb = np.zeros((self.ArcNum, self.DestNum), dtype=float)
        self.WaitArc = [[-1 for j in range(len(self.SNodes))]
                        for i in range(len(self.Transits))]
        self.BoardArc = [[-1 for j in range(len(self.SNodes))]
                        for i in range(len(self.Transits))]
        for i in range(len(self.Arcs)):
            Arc = self.Arcs[i]
            if Arc.FType == 1:
                # Case 1: Boarding Demand
                self.BoardArc[Arc.FLine][self.Nodes[Arc.FOut].FPlatform] = i
            elif Arc.FType == 3:
                # Case 3: Stopping
                self.WaitArc[Arc.FLine][self.Nodes[Arc.FOut].FPlatform] = i

    #--------------------------------------------------------------------------
    def TransitUpdate(self, NewFreq, NewCapacity):
        """Updates transit data given new frequency and capacity vectors.

        Requires a Freq and a Capacity vector, both the same length as the line
        vector and arranged in the same order.

        This repeats some portions of the DataLoad() method related to reading
        data from the Transitdata.txt file.
        """

        # Update transit object frequency and capacity attributes
        for i in range(len(self.Transits)):
            self.Transits[i].update(NewFreq[i], NewCapacity[i])

        # Update line arc frequency attributes
        for a in self.Arcs:
            if a.FType == 1:
                a.FFreq = self.Transits[a.FLine].FFreq
                a.FStrength = 1/a.FFreq

    #--------------------------------------------------------------------------
    def Calculation(self, RandomStart=True):
        """The main driver of the CapCon algorithm.

        Reads in network data from the local input folder and then evaluates
        the CapCon assignment model to produce the travel volumes in the local
        output folder.

        Accepts the following optional keyword arguments:
            RandomStart -- Indicates whether to initialize with random failure
                probabilities. If False, the failure probabilities from the
                previous execution will be used. Defaults to True.
        """

        # Reset flow volume arrays
        self.DNodeVol = np.zeros_like(self.DNodeVol)
        self.ODNodeVol = np.zeros_like(self.ODNodeVol)
        self.DArcVol = np.zeros_like(self.DArcVol)
        self.DArcVol2 = np.zeros_like(self.DArcVol2)
        self.ODArcVol = np.zeros_like(self.ODArcVol)
        self.ArcVolOld = np.zeros_like(self.ArcVolOld)
        self.ArcVolNew = np.zeros_like(self.ArcVolNew)
        self.OriginArcVol = np.zeros_like(self.OriginArcVol)
        self.OriginArcVol2 = np.zeros_like(self.OriginArcVol2)
        self.TransProb = np.zeros_like(self.TransProb)

        # Initialize random failure probabilities
        if RandomStart == True:
            # If False, we will start with the previous failure probabilities
            for n in self.ENodes:
                n.FFail = np.random.rand()

        Itera = 0

        # Main loop
        while Itera < self.IteraMax:

            # Calculate failure arc transition probabilities
            for n in self.ENodes:
                for a in n.FOutArcs:
                    if a.FType == 5:
                        a.FStrength = n.FFail
                    else:
                        a.FStrength = 1 - n.FFail

            self.TransProb = np.zeros_like(self.TransProb)

            # Conduct hyperpath search
            self._HyperpathSearch()

            # Conduct Markovian loading assignment
            self._AssignTraffic()

            # Method of Successive Averages for arc volumes
            if Itera == 0:
                # Process during first iteration
                Itera += 1
                self.ArcVolOld[:] = self.ArcVolNew[:]
                self.ArcVolNew = np.zeros_like(self.ArcVolNew)
                self.OriginArcVol[:] = self.OriginArcVol2[:]
                self.OriginArcVol2 = np.zeros_like(self.OriginArcVol2)
                self.DArcVol2 += self.DArcVol
            else:
                # Process after first iteration
                Itera += 1
                self.ArcVolOld = ((1-1/Itera)*self.ArcVolOld +
                                  (1/Itera)*self.ArcVolNew)
                self.ArcVolNew = np.zeros_like(self.ArcVolNew)
                self.OriginArcVol = ((1-1/Itera)*self.OriginArcVol +
                                     (1/Itera)*self.OriginArcVol2)
                self.OriginArcVol2 = np.zeros_like(self.OriginArcVol2)
                self.DArcVol2 = ((1-1/Itera)*self.DArcVol2 +
                                 (1/Itera)*self.DArcVol)

            # Update fail-to-board probabilities based on arc volumes
            MaxFFailDiff = 0 # maximum change in failure probabilities
            for n in self.Nodes:
                if n.FType != 5:
                    # Only proceed for failure nodes
                    continue
                # Find indices of wait/board arcs at failure node
                WaitArcNo = self.WaitArc[n.FLine][n.FPlatform]
                BoardArcNo = self.BoardArc[n.FLine][n.FPlatform]

                if WaitArcNo != -1:
                    WaitArcVol = self.ArcVolOld[WaitArcNo]
                else:
                    WaitArcVol = 0.0
                BoardArcVol = self.ArcVolOld[BoardArcNo]

                AvailableCapa = self.Transits[n.FLine].FCapacity - WaitArcVol

                if BoardArcVol != 0:
                    MaxFFailDiff = max(MaxFFailDiff, abs(n.FFail - min(1,
                                       max(0, 1-AvailableCapa/BoardArcVol))))
                    n.FFail = min(1, max(0, 1-AvailableCapa/BoardArcVol))
                else:
                    MaxFFailDiff = max(MaxFFailDiff, n.FFail)
                    n.FFail = 0.0

            if ((MaxFFailDiff <= self.ErrorThreshold)
                and (Itera > self.IteraMin)):
                # If error is small enough, immediately go to last iteration
                Itera = self.IteraMax

    #--------------------------------------------------------------------------
    def CalculateFlows(self, RandomStart=True):
        """Calls Calculation() method and returns final flow vector."""

        self.Calculation(RandomStart=RandomStart)

        return self.ArcVolOld

    #--------------------------------------------------------------------------
    def _CalcCost(self, Node, ItemNo):
        """Node cost calculation subroutine for use in the hyperpath search.

        Note that in the original Pascal program this was a method of the TNode
        class. However, since the method requires access to the full node list
        of the CapCon object, we have instead made it a method of the CapCon
        class and added an argument to specify a TNode object.
        """

        F = 0.0
        for i in range(ItemNo+1):
            Arc1 = Node.FActiveOutArcs[i]
            F += 1/Arc1.FFreq

        Result = 1.0

        for i in range(ItemNo+1):
            Arc1 = Node.FActiveOutArcs[i]
            Node1 = self.Nodes[Arc1.FIn]
            Result += 1/Arc1.FFreq*(Arc1.FCost+Node1.FCost)

        return Result/F

    #--------------------------------------------------------------------------
    def _HyperpathSearch(self):
        """Minimum-cost hyperpath search subroutine.

        This is the dynamic programming algorithm for finding the minimum-cost
        hyperpath to each destination node. It is called as a subroutine of the
        Calculation method.
        """

        NodeCost = [0.0 for i in range(self.NodeNum)]

        #......................................................................
        # Outer loop begin (find a hyperpath for each destination)
        # Lines 620 to 961 of Pascal code
        for DNum in range(self.DestNum):
            # Initialization
            N1 = []
            N2 = []
            N3 = []
            M1 = []
            M2 = []

            DestNode = self.DNodes[DNum] # Current hyperpath destination

            for n in self.Nodes:
                if DestNode.FID == n.FID:
                    N3.append(n) # Adding Dest node to selected nodes
                    n.FCost = 0
                else:
                    N1.append(n) # Add all nodes to pool N1
                    n.FCost = self.Infinity

            for a in self.Arcs:
                if a.FType == 5:
                    # Failing arc
                    a.FCost = self.Infinity
                else:
                    a.FCost = a.FTrTime

            for a in self.Arcs:
                M1.append(a) # Add all arcs to pool M1
                a.FChecked = False

            for i in range(self.NodeNum):
                NodeCost[i] = self.Infinity

            #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            # Inner loop begin (finding node of minimum cost to destination)
            # Lines 662-910 of Pascal code
            while len(N3) < self.NodeNum:

                # Initialization
                MinCost = self.Infinity
                MinArcIndex = -1
                for n in self.Nodes:
                    n.FActiveOutArcs.clear()

                # Process N3
                for n in N3:
                    for a in n.FInArcs:
                        if a.FChecked:
                            # Skip to next loop if already checked
                            continue
                        if n.FCost + a.FCost <= MinCost:
                            MinCost = n.FCost + a.FCost
                            MinArcIndex = a.FID

                # Process N2
                for n in N2:
                    for a in n.FInArcs:
                        if a.FChecked:
                            # Skip to next loop if already checked
                            continue
                        if n.FCost + a.FCost <= MinCost:
                            MinCost = n.FCost + a.FCost
                            MinArcIndex = a.FID

                if MinArcIndex == -1:
                    # Break out of loop if MinCost has not been improved
                    break
                Arc1 = self.Arcs[MinArcIndex]
                Arc1.FChecked = True
                Node1 = self.Nodes[Arc1.FOut]

                # Update costs
                if Node1.FType == 2:
                    # Case 2: Stopping Node, lines 708-779

                    # Create temporal list
                    for a in Node1.FOutArcs:
                        if M1.count(a) == 0:
                            # Arc not in M1
                            Node1.FActiveOutArcs.append(a)
                    Node1.FActiveOutArcs.append(Arc1)

                    # Sort active out arcs (in ascending order of cost)
                    for i in range(len(Node1.FActiveOutArcs)-1):
                        for j in range(i+1, len(Node1.FActiveOutArcs)):
                            Arc1 = Node1.FActiveOutArcs[i]
                            Arc2 = Node1.FActiveOutArcs[j]
                            Node2 = self.Nodes[Arc1.FIn]
                            Cost1 = Arc1.FCost + Arc1.FFreq + Node2.FCost
                            Node2 = self.Nodes[Arc2.FIn]
                            Cost2 = Arc2.FCost + Arc2.FFreq + Node2.FCost
                            if Cost2 < Cost1:
                                (Node1.FActiveOutArcs[i],
                                     Node1.FActiveOutArcs[j]) = (
                                     Node1.FActiveOutArcs[j],
                                      Node1.FActiveOutArcs[i])

                    # Find an arc set of minimum cost
                    Arc1 = Node1.FActiveOutArcs[0]
                    Node2 = self.Nodes[Arc1.FIn]
                    Cost1 = Node2.FCost + Arc1.FCost + Arc1.FFreq

                    cs = 1 # Number of arcs chosen for set K
                    for i in range(1, len(Node1.FActiveOutArcs)):
                        # With common lines
                        Cost2 = self._CalcCost(Node1, i)
                        if Cost2 > Cost1:
                            break
                        Cost1 = Cost2
                        cs += 1

                    # Update M1 and M2
                    if Node1.FCost >= Cost1:
                        # Cost improvement
                        Node1.FCost = Cost1
                        NodeCost[Node1.FID] = Node1.FCost
                        for j in range(len(Node1.FActiveOutArcs)):
                            # Note: cs = number of active out arcs
                            if j < cs:
                                Arc1 = Node1.FActiveOutArcs[j]
                                if M1.count(Arc1) > 0:
                                    # Arc1 present in M1
                                    M1.remove(Arc1)
                                if M2.count(Arc1) == 0:
                                    # Arc1 absent from M2
                                    M2.append(Arc1)
                            else:
                                Arc1 = Node1.FActiveOutArcs[j]
                                if M1.count(Arc1) == 0:
                                    # Arc1 absent from M1
                                    M1.append(Arc1)
                                if M2.count(Arc1) > 0:
                                    # Arc1 present in M2
                                    M2.remove(Arc1)

                elif Node1.FType == 5:
                    # Case 5: Failure Node, lines 780-791
                    if Node1.FFail == 1:
                        MinCost = 0.1*self.Infinity
                    else:
                        MinCost -= self.RiskParameter * np.log(1-Node1.FFail)
                    M1.remove(Arc1)
                    M2.append(Arc1)
                    Node1.FCost = MinCost
                    NodeCost[Node1.FID] = Node1.FCost

                else:
                    # All other cases (boarding, alighting, origin,
                    # destination), lines 792-813
                    if Node1.FCost >= MinCost:
                        for a in Node1.FOutArcs:
                            if M1.count(a) == 0:
                                # Arc absent from M1
                                M1.append(a)
                            if M2.count(a) > 0:
                                # Arc present in M2
                                M2.remove(a)
                        M1.remove(Arc1)
                        M2.append(Arc1)
                        Node1.FCost = MinCost
                        NodeCost[Node1.FID] = Node1.FCost

                # Special procedures for boarding nodes
                if Node1.FType == 3:
                    Cost3 = Node1.FCost
                    for n in self.Nodes:
                        if (n.FType == 2) and (n.FPlatform == Node1.FPlatform):
                            # Stop node that shares a platform with Node1
                            Node6 = n

                    for a in Node6.FInArcs:
                        Arc2 = a
                        if Arc2.FType == 2:
                            # Alighting arc
                            Node2 = self.Nodes[a.FOut] # Alighting node
                            for a2 in Node2.FInArcs:
                                Arc3 = a2 # Line arc
                            Node3 = self.Nodes[Arc3.FOut] # Boarding node
                            for a2 in Node2.FOutArcs:
                                Arc6 = a2 # Stopping arc
                                if Arc6.FType == 3:
                                    break

                            for a2 in Node3.FInArcs:
                                Arc4 = a2 # Boarding arc
                                if Arc4.FType == 6:
                                    break
                            Node4 = self.Nodes[Arc4.FOut] # Failure node

                            for a2 in Node4.FInArcs:
                                Arc5 = a2 # Boarding demand arc
                            Node5 = self.Nodes[Arc5.FOut] # Stop node

                            Cost3 += (Arc2.FCost + Arc3.FCost + Arc4.FCost
                                + Arc5.FCost)

                            if NodeCost[Node5.FID] < Cost3:
                                Arc2.FCost = self.Infinity
                                Arc3.FCost = self.Infinity
                                Arc4.FCost = self.Infinity
                                Arc5.FCost = self.Infinity
                                Arc6.FCost = self.Infinity

                # Update N1 and N2
                CheckNode = True
                for a in Node1.FOutArcs:
                    if a.FChecked == False:
                        CheckNode = False
                        break

                if CheckNode == True:
                    if N3.count(Node1) == 0:
                        # Node1 absent from N3
                        N3.append(Node1)
                    if N1.count(Node1) > 0:
                        # Node1 present in N1
                        N1.remove(Node1)
                    if N2.count(Node1) > 0:
                        # Node1 present in N2
                        N2.remove(Node1)
                else:
                    if N1.count(Node1) > 0:
                        # Node1 present in N1
                        N1.remove(Node1)
                    if N2.count(Node1) == 0:
                        # Node1 absent from N2
                        N2.append(Node1)

                # Update failing arc
                if Node1.FType == 5:
                    for a in Node1.FOutArcs:
                        Arc1 = a
                        if Arc1.FType == 5:
                            break

                    if Node1.FFail == 1:
                        Arc1.FCost = 0.1*self.Infinity
                    else:
                        Arc1.FCost = (Node1.FCost +
                                      self.RiskParameter*np.log(1-Node1.FFail))

            #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            # Inner loop end

            # Updating transition probabilities
            for n in self.Nodes:
                if n.FType != 5:
                    Total = 0
                    for a in n.FOutArcs:
                        if M2.count(a) > 0:
                            # Arc present in M2
                            Total += a.FStrength

                    for a in n.FOutArcs:
                        if M2.count(a) > 0:
                            # Arc present in M2
                            if Total != 0:
                                self.TransProb[a.FID][DNum] = (
                                    a.FStrength/Total)
                            else:
                                self.TransProb[a.FID][DNum] = 0.0

                else:
                    for a in n.FOutArcs:
                        if M2.count(a) > 0:
                            # Arc present in M2
                            if a.FType == 6 or a.FType == 5:
                                self.TransProb[a.FID][DNum] = a.FStrength

        #......................................................................
        # Outer loop end

    #--------------------------------------------------------------------------
    def _AssignTraffic(self):
        """Traffic assignment subroutine.

        This is the Markovian loading process for finding the arc traffic flow
        values based on the transition probabilities. It is called as a
        subroutine of the Calculation method.
        """

        # Initialization
        Q1 = np.zeros((self.OriginNum, self.NodeNum-self.OriginNum-1),
                           dtype=float)
        Q2 = np.zeros((self.NodeNum-self.OriginNum-1,
                            self.NodeNum-self.OriginNum-1), dtype=float)
        Q3 = np.zeros((self.OriginNum, self.NodeNum-self.OriginNum-1),
                           dtype=float)
        NodeSeq = [0 for i in range(self.NodeNum-1)]
        NodeSeq2 = [0 for i in range(self.NodeNum)]

        # Destination loop
        for s in range(self.DestNum):

            DestNode = self.DNodes[s]
            NodeNo = 0

            for i in range(self.NodeNum):
                if self.Nodes[i].FID == DestNode.FID:
                    # Skip current destination
                    continue
                if self.Nodes[i].FType == 0:
                    # Origin node to place in sequence
                    NodeSeq[NodeNo] = i
                    NodeNo += 1

            for i in range(self.NodeNum):
                if self.Nodes[i].FID == DestNode.FID:
                    # Skip current destination
                    continue
                if self.Nodes[i].FType != 0:
                    # Non-origin node to place in sequence
                    NodeSeq[NodeNo] = i
                    NodeNo += 1

            NodeSeq2 = [-1 for i in range(self.NodeNum)]
            for i in range(self.NodeNum-1):
                NodeSeq2[NodeSeq[i]] = i # last element is -1

            # Create Q1
            Q1 = np.zeros_like(Q1)
            for i in range(self.OriginNum):
                for Arc in self.Nodes[NodeSeq[i]].FOutArcs:
                    Q1[i][NodeSeq2[Arc.FIn]-self.OriginNum] = (
                        self.TransProb[Arc.FID][s])

            # Create Q2
            Q2 = np.zeros_like(Q2)
            for i in range(self.NodeNum-self.OriginNum-1):
                Node1 = self.Nodes[NodeSeq[i+self.OriginNum]]
                for Arc in Node1.FOutArcs:
                    if NodeSeq2[Arc.FIn] != -1:
                        Q2[i][NodeSeq2[Arc.FIn]-self.OriginNum] = (
                            self.TransProb[Arc.FID][s])

            # Calculate (I-Q)^-1
            Q2 = np.linalg.inv(np.eye(self.NodeNum-self.OriginNum-1) - Q2)

            Q3 = np.zeros_like(Q3)
            for i in range(self.OriginNum):
                for j in range(self.NodeNum-self.OriginNum-1):
                    for k in range(self.NodeNum-self.OriginNum-1):
                        Q3[i][j] += Q1[i][k]*Q2[k][j]

            for j in range(self.NodeNum-self.OriginNum-1):
                self.DNodeVol[j][s] = 0.0
                for k in range(self.OriginNum):
                    self.ODNodeVol[j][k][s] = 0.0
                for k in range(self.OriginNum):
                    self.DNodeVol[j][s] += self.OD[k][s]*Q3[k][j]
                    self.ODNodeVol[j][k][s] += self.OD[k][s]*Q3[k][j]

            for i in range(self.ArcNum):
                Arc = self.Arcs[i]
                if self.Nodes[Arc.FOut].FType == 0:
                    # Origin node
                    self.DArcVol[i][s] = (
                        self.OD[self.Node2Origin[Arc.FOut]][s]
                        *self.TransProb[Arc.FID][s])
                    for k in range(self.OriginNum):
                        if k == self.Node2Origin[Arc.FOut]:
                            self.ODArcVol[i][k][s] = (self.OD[k][s]
                                *self.TransProb[Arc.FID][s])
                        else:
                            self.ODArcVol[i][k][s] = 0.0
                else:
                    # Non-origin node
                    self.DArcVol[i][s] = (self.DNodeVol[NodeSeq2[Arc.FOut]
                        -self.OriginNum][s]*self.TransProb[Arc.FID][s])
                    for k in range(self.OriginNum):
                        self.ODArcVol[i][k][s] = (
                            self.ODNodeVol[NodeSeq2[Arc.FOut]
                            -self.OriginNum][k][s]*self.TransProb[Arc.FID][s])

        # End of destination loop

        for i in range(self.ArcNum):
            for s in range(self.DestNum):
                self.ArcVolNew[i] += self.DArcVol[i][s]

        for i in range(self.ArcNum):
            for k in range(self.OriginNum):
                for s in range(self.DestNum):
                    self.OriginArcVol2[i][k] += self.ODArcVol[i][k][s]
