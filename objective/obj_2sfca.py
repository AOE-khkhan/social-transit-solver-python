"""A module for calculating the 2SFCA-based objective.

The main public class, Objective, includes a variety of methods for calculating
two-step floating catchment area (2SFCA) metrics of a set of population
centers. Upon initialization it reads in network data and maintains this
information along with a CPLEX object to recalculate the 2SFCA metrics during
each iteration of the search process.

There are also private classes for storing node and arc data.
"""

import cplex as cp
import operator as op

#==============================================================================
class Objective:
    """A class for calculating the 2SFCA-based objective.

    The object created from this class should be initialized with some basic
    information about the underlying network, and should be updated each
    iteration depending on the current solution. Its main public method uses
    its internal information to output the objective value.
    """

    #--------------------------------------------------------------------------
    def __init__(self, data="", logfile="metrics.txt"):
        """Objective object constructor.

        Automatically reads in the network data and initializes the CPLEX
        solver for use in calculating shortest paths.

        Accepts the following optional keyword arguments:
            data -- Root directory containing the network information. Defaults
                to the current working directory.
            logfile -- Output file path. The 2SFCA metric for each population
                center is written to this file at the end of the search.
                Defaults to the current working directory.
            turnaround -- Total time required for each route to reset between
                runs (in minutes). This may include a combination of layover
                time and driving back to the route start.
        """

        self.data = data
        self.logfile = logfile

        # Load node, arc, and misc data
        self._load_nodes()
        self._load_arcs()
        self._load_misc()

        # Define Cplex object to define shortest path LP
        self._cplex_setup()

    #--------------------------------------------------------------------------
    def __del__(self):
        """Objective object destructor. Closes the CPLEX solver."""

        self.lp.end()

    #--------------------------------------------------------------------------
    def _load_nodes(self):
        """Loads node data from data file.

        Creates a list of node objects, sets their internal attributes, and
        creates lists of specific node types for use in interacting with the
        CPLEX solver.
        """

        # Initialize node lists
        self.nodes = [] # node objects for every node
        self.origins = [] # node objects for origins only
        self.destinations = [] # node objects for destinations only

        # Populate node lists
        i = 0
        with open(self.data+"Nodedata_Obj.txt", 'r') as f:
            for line in f:
                i += 1
                if i > 1:
                    # Skip comment line
                    dum = line.split()
                    # Initialize a new node
                    self.nodes.append(_Node(NodeID=int(dum[0]),
                                            NodeType=int(dum[1]),
                                            Line=int(dum[2]),
                                            Val=float(dum[3])))
                    if int(dum[1]) == 0:
                        # Add a new origin node to the origin list
                        self.origins.append(self.nodes[-1])
                    if int(dum[1]) == 1:
                        # Add a new destination node to the destination list
                        self.destinations.append(self.nodes[-1])

        # Set variable name and distance lists for origins and destinations
        for s in self.origins:
            s.var = [self._var_name(n.index) for n in self.destinations]
            s.dist = [1.0 for n in self.destinations]
        for s in self.destinations:
            s.var = [self._var_name(n.index) for n in self.origins]
            s.dist = [1.0 for n in self.origins]

    #--------------------------------------------------------------------------
    def _load_arcs(self):
        """Loads arc data from data file.

        Creates a list of arc objects, sets their internal attributes, and
        creates lists of specific arc types for use in interacting with the
        CPLEX solver.

        We also calculate the total travel time for a complete circuit of each
        line, which is required for calculating the average waiting time for a
        given solution.
        """

        # Initialize arc and line lists
        self.arcs = [] # arc objects for every arc
        turnaround = [] # turnaround time for each line

        # Populate line lists
        i = 0
        with open(self.data+"Transitdata.txt", 'r') as f:
            for line in f:
                i += 1
                if i > 1:
                    # Skip comment line
                    dum = line.split()
                    turnaround.append(float(dum[5]))

        # Populate arc lists
        i = 0
        with open(self.data+"Arcdata_Obj.txt", 'r') as f:
            for line in f:
                i += 1
                if i > 1:
                    # Skip comment line
                    dum = line.split()
                    # Initialize a new arc
                    self.arcs.append(_Arc(ArcID=int(dum[0]),
                                          ArcType=int(dum[1]),
                                          Line=int(dum[2]),
                                          Out=int(dum[3]),
                                          In=int(dum[4]),
                                          TrTime=float(dum[5])))

        # Create line boarding arc list
        self.line_boarding = [[] for i in range(len(turnaround))]
        for a in self.arcs:
            if a.type == 1:
                self.line_boarding[a.line].append(a)

        # Calculate list of total circuit times for each line
        self.line_time = turnaround[:]
        for a in self.arcs:
            if a.line >= 0:
                self.line_time[a.line] += a.cost # add line arc cost to total

    #--------------------------------------------------------------------------
    def _load_misc(self):
        """Loads problem parameters from data file.

        This includes parameters for the CPLEX object and the 2SFCA metric
        calculation.
        """

        i = -1
        with open(self.data+"Problem_Data.txt", 'r') as f:
            for i in range(15):
                # Skip past the unneeded parameters
                f.readline()
            self.epsilon = float(f.readline()) # CPLEX cleanup epsilon
            self.time_cutoff = float(f.readline()) # 2SFCA time cutoff

    #--------------------------------------------------------------------------
    def _cplex_setup(self):
        """Initializes the Cplex object and defines the shortest path LP.

        The LP here concerns a network G = (V,E) for which we want to calculate
        all pairwise distances from one given node s to every node in a subset
        S of V. The LP takes the following form:

            max  sum_{t in S} x_t
            s.t. x_j - x_i <= c_ij     for all ij in E
                 x_s = 0               for all s in S

        After solving this LP, x_t will represent the distance from s to t for
        every node t in S. Note that x_i will not in general represent the
        correct distances for nodes i not in S.
        """

        # Initialize Cplex object
        self.lp = cp.Cplex()

        # Silence the CPLEX output streams
        self.lp.set_log_stream(None)
        self.lp.set_results_stream(None)
        self.lp.set_error_stream(None)
        self.lp.set_warning_stream(None)

        # Set LP as maximization
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

        # Variable names correspond to every node
        var_names = [self._var_name(n.index) for n in self.nodes]

        # Objective coefficients will be 1 for nodes in the opposite origin/
        # destination set from the source and 0 otherwise; initialize all as 0
        var_obj = [0 for i in range(len(var_names))]

        # Source variable is zero while all others are free; init all as free
        var_lb = [-cp.infinity for i in range(len(var_names))]
        var_ub = [cp.infinity for i in range(len(var_names))]

        # Define LP variables, bounds, and objective coefficients
        self.lp.variables.add(names=var_names, obj=var_obj, lb=var_lb,
                              ub=var_ub)

        # Constraint names correspond to every arc
        con_names = [self._con_name(a.index) for a in self.arcs]

        # All constraints have sense 'L' (for <=)
        con_senses = "L"*len(con_names)

        # Constraint RHS constant is the associated arc's cost
        con_rhs = [a.cost for a in self.arcs]

        # Each element of the constraint linear expression is a list of two
        # lists, the first of which is a set of variable names and the second
        # of which is the set of corresponding constraints
        con_lin_expr = [[[self._var_name(a.head), self._var_name(a.tail)],
                          [1, -1]] for a in self.arcs]

        # Define LP constraints
        self.lp.linear_constraints.add(names=con_names, lin_expr=con_lin_expr,
                                       senses=con_senses, rhs=con_rhs)

    #--------------------------------------------------------------------------
    def _calculate_distances(self):
        """Uses the shortest path LP to find all pairwise O/D distances.

        The LP is set up to calculate all distances from a single source to
        everything in a destination set. For this reason we will need to loop
        through every origin and every destination, setting up the
        corresponding LP for each and re-solving based on the updated
        parameters.
        """

        # Find all origin-to-destination distances

        # Set all origin objective coefficients to 0
        new_coef = [(self._var_name(n.index), 0) for n in self.origins]
        self.lp.objective.set_linear(new_coef)

        # Set all destination objective coefficients to 1
        new_coef = [(self._var_name(n.index), 1) for n in self.destinations]
        self.lp.objective.set_linear(new_coef)

        # Main origin loop
        for s in self.origins:
            # Free all origin and destination variables
            new_bound = [(self._var_name(n.index), -cp.infinity) for n in
                         self.origins+self.destinations]
            self.lp.variables.set_lower_bounds(new_bound)
            new_bound = [(self._var_name(n.index), cp.infinity) for n in
                         self.origins+self.destinations]
            self.lp.variables.set_upper_bounds(new_bound)

            # Set current source's variable to 0
            self.lp.variables.set_lower_bounds(self._var_name(s.index), 0)
            self.lp.variables.set_upper_bounds(self._var_name(s.index), 0)

            # Solve model
            self.lp.solve()

            # Read results from model into node distance list
            s.dist = self.lp.solution.get_values([self._var_name(n.index) for
                                                  n in self.destinations])

        # Find all destination-to-origin distances

        # Set all destination objective coefficients to 0
        new_coef = [(self._var_name(n.index), 0) for n in self.destinations]
        self.lp.objective.set_linear(new_coef)

        # Set all origin objective coefficients to 1
        new_coef = [(self._var_name(n.index), 1) for n in self.origins]
        self.lp.objective.set_linear(new_coef)

        # Main destination loop
        for s in self.destinations:
            # Free all origin and destination variables
            new_bound = [(self._var_name(n.index), -cp.infinity) for n in
                         self.origins+self.destinations]
            self.lp.variables.set_lower_bounds(new_bound)
            new_bound = [(self._var_name(n.index), cp.infinity) for n in
                         self.origins+self.destinations]
            self.lp.variables.set_upper_bounds(new_bound)

            # Set current source's variable to 0
            self.lp.variables.set_lower_bounds(self._var_name(s.index), 0)
            self.lp.variables.set_upper_bounds(self._var_name(s.index), 0)

            # Solve model
            self.lp.solve()

            # Read results from model into node distance list
            s.dist = self.lp.solution.get_values([self._var_name(n.index) for
                                                  n in self.origins])

    #--------------------------------------------------------------------------
    def _var_name(self, i):
        """Template for generating variable names given a node ID."""

        return "x("+str(i)+")"

    #--------------------------------------------------------------------------
    def _con_name(self, i):
        """Template for generating constraint names given an arc ID."""

        return "c("+str(i)+")"

    #--------------------------------------------------------------------------
    def _update_costs(self, sol):
        """Updates the Cplex object costs for a new solution vector.

        The solution vector determines the average headway for each line, which
        in turn determines the cost of each arc. The update process requires us
        to calculate the new headways and then change the corresponding RHS
        constants in the LP.
        """

        self._waiting_time(sol) # update all necessary arc costs

        for i in range(len(sol)):
            for a in self.line_boarding[i]:
                # RHS update requires a list of (name, value) pairs
                self.lp.linear_constraints.set_rhs(self._con_name(a.index),
                                                   a.cost)

    #--------------------------------------------------------------------------
    def _waiting_time(self, sol):
        """Calculates average headway from a solution and updates arcs."""

        for i in range(len(sol)):
            for a in self.line_boarding[i]:
                if sol[i] > 0:
                    # Average headway is total time per total vehicles
                    a.cost = self.line_time[i] / sol[i]
                else:
                    # Zero vehicles means infinite headway
                    a.cost = cp.infinity

    #--------------------------------------------------------------------------
    def _fca(self):
        """Conducts and returns the 2SFCA calculations for all origin nodes.

        In order to calculate the 2SFCA metric for each population center i, we
        first calculate the physician-to-population ratio for each facility j
        as

            R_j = S_j / (sum_{k : d_kj <= d_0} P_k)

        Here, S_j is the number of physicians at j (or some other metric of
        capacity or quality), P_k is the population at center k, d_kj is the
        distance from k to j, and d_0 is a travel time cutoff that defines the
        catchment areas. The ratio R_j is then the average number of
        physicians per capita within the distance cutoff.

        The accessibility metric for center i is then

            A_i = sum_{j : d_ij <= d_0} R_j

        This is the sum of physician-to-population ratios of all facilities for
        which the community is within the catchment area.
        """

        # Calculate physician-to-population ratio for all destinations
        phys_to_pop = [0 for i in range(len(self.destinations))]
        for j in range(len(phys_to_pop)):
            phys = self.destinations[j].val # number of physicians
            pop = 0 # running total of population
            for n in self.origins:
                if n.dist[j] <= self.time_cutoff:
                    # Only include population within catchment area
                    pop += n.val
            if pop > 0:
                # Only update if catchment area is nonempty
                phys_to_pop[j] = phys/pop

        # Calculate total phys-to-pop ratio for all origins
        access = [0 for i in range(len(self.origins))]
        for i in range(len(access)):
            for j in range(len(phys_to_pop)):
                if self.origins[i].dist[j] <= self.time_cutoff:
                    # Only include facilities within catchment area
                    access[i] += phys_to_pop[j]

        return access

    #--------------------------------------------------------------------------
    def _gravity(self, node):
        """Calculates the gravity-based accessibility metric of a pop center.

        Returns the a gravity-based accessibility metric for a specified origin
        node list index. This metric can be interpreted as a continuous analog
        of the 2SFCA metric.

        This is meant for use as a secondary objective for the purposes of
        breaking ties during the neighborhood search. It is likely that many
        neighbors will be tied for the minimum 2SFCA metric, but we would still
        like to preferentially choose moves likely to improve this metric.

        Simply using the 2SFCA metric of the least accessible community will
        not help because moves that improve it will already show an objective
        improvement, eliminating the need to break the tie. In addition, the
        2SFCA metric is inherently discontinuous and likely not to change very
        much within large regions of the search space. The gravity metric,
        however, is continuous, and should be different for every solution. It
        is also similar to the 2SFCA metric in that it is higher for a
        community that is near a lot of facilities with excess capacity.

        Specifically, we begin by calculating a continuous analog of the 2SFCA
        metric's physician-to-population ratio as

            R_j = S_j * sum_k (P_k/d_kj)

        The sum is now over all population centers, but each population is
        divided by its distance, making more distant populations count less.
        The continuous analog of the 2SFCA metric for population center i is
        then

            A_i = sum_j (R_j/d_ij)

        Again, the sum is now over all facilities, but more distant facilities
        count less.
        """

        # Calculate physician-to-population ratio for all destinations
        phys_to_pop = [0 for i in range(len(self.destinations))]
        for j in range(len(phys_to_pop)):
            for n in self.origins:
                # Increment by ratio of population to distance
                phys_to_pop[j] += n.val/n.dist[j]
            # Multiply ratio sum by staff size
            phys_to_pop[j] *= self.destinations[j].val

        # Calculate sum of distance-adjusted phys-to-pop ratios for target node
        access = 0
        for j in range(len(phys_to_pop)):
            # Increment by ratio of facility metric to distance
            access += phys_to_pop[j]/self.destinations[j].dist[node]

        return access

    #--------------------------------------------------------------------------
    def calculate(self, sol):
        """Calculates the objective value for a given solution.

        We output a tuple of the minimum 2SFCA metric (which is meant to be our
        real objective) and the gravity-based metric of the community with the
        minimum 2SFCA metric, for use as a tiebreaker.

        The objective being considered in this problem is to maximize the
        minimum 2SFCA metric over all population centers. However, since the
        main driver is set up to attempt to minimize its objective, this
        method will actually return the negative of the minimum 2SFCA metric.
        Likewise, the second output is the negative gravity metric.
        """

        met = self.metrics(sol) # list of all metrics

        # Find the origin with the lowest 2SFCA metric
        min_index, min_metric = min(enumerate(met), key=op.itemgetter(1))

        return -min_metric, -self._gravity(min_index)

    #--------------------------------------------------------------------------
    def metrics(self, sol):
        """Calculates a list of 2SFCA metrics resulting from a given solution.

        The metrics are returned as a list with the same order as the
        population center IDs in the Nodedata_Obj.txt file. This method is
        called directly for the final solution to be able to output the final
        list of 2SFCA metrics, but it is also used in the calculate() method to
        calculate the objective value.
        """

        self.lp.cleanup(self.epsilon) # clean up solver leftovers
        self._update_costs(sol) # update the Cplex object
        self._calculate_distances() # calculate the O/D pair distances

        return self._fca() # carry out and return the 2SFCA calculation

    #--------------------------------------------------------------------------
    def output(self, sol):
        """Prints the list of all 2SFCA metrics for a solution to a file.

        This should be used to calculate the 2SFCA metrics for the original
        network and then again for the final network after the solution
        algorithm has ended. As part of the results analysis we can compare the
        two to see how much each area has improved.
        """

        met = self.metrics(sol) # calculate metrics

        with open(self.logfile, 'w') as f:
            print("node\tmetric", file=f) # print comment line
            for i in range(len(met)):
                print(str(self.origins[i].index)+"\t"+str(met[i]), file=f)

#==============================================================================
class _Node:
    """A class for storing node-level attributes."""

    #--------------------------------------------------------------------------
    def __init__(self, NodeID=-1, NodeType=-1, Line=-1, Val=-1):
        """Node object constructor. Initializes node-level attributes.

        Accepts the following optional keyword arguments:
            NodeID -- Node index.
            NodeType -- Node type index.
                0: origin
                1: destination
                2: platform
                3: line
            Line -- Index of associated line.
            Val -- Associated value (population at origin, capacity at
                destination).
        """

        self.index = NodeID
        self.type = NodeType
        self.line = Line
        self.val = Val

        # If this node is an origin or destination, the following list will
        # contain the distances to every node in the opposite set, ordered in
        # the same way as the main origin and destination node lists.
        self.dist = []

#==============================================================================
class _Arc:
    """A class for storing arc-level attributes."""

    #--------------------------------------------------------------------------
    def __init__(self, ArcID=-1, ArcType=-1, Line=-1, Out=-1, In=-1,
                 TrTime=-1):
        """Arc object constructor. Initializes arc-level attributes.

        Accepts the following optional keyword arguments:
            ArcID -- Arc index.
            ArcType -- Arc type index.
                0:line
                1:boarding
               -1:other
            Line -- Index of associated line.
            Out -- Index of tail node (node which this arc leaves).
            In -- Index of head node (node which this arc enters).
            TrTime -- Travel time for arc.
        """

        self.index = ArcID
        self.type = ArcType
        self.line = Line
        self.tail = Out
        self.head = In
        self.cost = TrTime
