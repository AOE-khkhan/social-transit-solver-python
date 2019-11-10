"""A module for calculating the constraint function values.

The main public class, Constraint, stores information to define the constraint
functions and contains methods to calculate the constraint function values for
a given solution vector. It makes use of the Spiess module to solve the
assignment problem for a given solution vector.
"""

# Note: This import path assumes that this module is being called from within
# the main driver module.
import constraints.assignment.spiess as sp

#==============================================================================
class Constraint:
    """A class meant for calculating the operator and user costs.

    The object created from this class should be initialized with some basic
    information about the underlying network, and should be updated each
    iteration depending on the current solution. It includes public methods for
    calculating various constraint function values.
    """

    #--------------------------------------------------------------------------
    def __init__(self, data="", operator_percent=0.01, user_percent=0.01):
        """Constraint object constructor.

        Accepts the following optional keyword arguments:
            data -- Root directory containing the network information. Defaults
                to the current working directory.
            operator_percent -- Allowable percentage increase in the operator
                cost. Defaults to 0.01.
            user_percent -- Allowable percentage increase in the user cost.
                Defaults to 0.01.
        """

        self.data = data
        self.operator_percent = operator_percent
        self.user_percent = user_percent

        # First-time setup related to arc and line structures
        self._setup()

        # Initialize Spiess object
        self.spiess = sp.Spiess(data=self.data,
                                cplex_epsilon=self.cplex_epsilon,
                                optimality_epsilon=self.spiess_epsilon,
                                max_iterations=self.spiess_max)

    #--------------------------------------------------------------------------
    def __del__(self):
        """Constraint object destructor. Deletes assignment model object."""

        del self.spiess

    #--------------------------------------------------------------------------
    def _setup(self):
        """Sets up a variety of internal data structures based on arc data.

        This method reads the arc data, transit data, and problem data files to
        initialize the internal data structures needed for the module. This
        includes calculating the total circuit time for each route (which is
        needed to calculate frequencies) and a list of boarding arcs for each
        line (which is needed for updating boarding arc attributes).
        """

        # Initialize line attribute lists
        self.line_time = [] # total circuit time
        self.line_freq = [] # line frequency (vehicles/minute)
        self.line_cap = [] # line capacity (total people)
        self.vtype = [] # line vehicle type

        # Read transit data
        i = -1
        with open(self.data+"Transitdata.txt", 'r') as f:
            for line in f:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()
                    self.line_freq.append(float(dum[1]))
                    self.line_cap.append(float(dum[2]))
                    self.line_time.append(float(dum[5])) # turnaround time
                    self.vtype.append(int(dum[7]))

        # Read problem data
        i = -1
        with open(self.data+"Problem_Data.txt", 'r') as f:
            f.readline() # comment line
            f.readline() # vehicle bounds
            self.vcapacity = [int(n) for n in f.readline().split()] # capacity
            self.vcost = [float(n) for n in f.readline().split()] # oper cost
            self.vfare = [float(n) for n in f.readline().split()] # fare
            self.operator_init = float(f.readline()) # initial operator cost
            self.user_init = float(f.readline()) # initial user cost
            f.readline() # constraint vector length
            self.riding_weight = float(f.readline()) # in-vehicle time weight
            self.walking_weight = float(f.readline()) # walking time weight
            self.waiting_weight = float(f.readline()) # waiting time weight
            self.bpr_power = float(f.readline()) # BPR function exponent
            self.time_horizon = float(f.readline()) # day-to-day time horizon
            self.spiess_epsilon = float(f.readline()) # Frank-Wolfe cutoff eps
            self.spiess_max = int(f.readline()) # Frank-Wolfe cutoff iterations
            self.cplex_epsilon = float(f.readline()) # CPLEX cleanup epsilon

        # List of boarding arc IDs associated with each line
        self.line_boarding = [[] for i in range(len(self.vtype))]

        # Initialize arc attribute lists
        self.arc_cost = [] # list of constant part of arc travel times
        self.arc_type = [] # list of type IDs of all arcs

        # Read arc data
        i = -1
        with open(self.data+"Arcdata_Con.txt", 'r') as f:
            for line in f:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()
                    self.line_time[int(dum[2])] += float(dum[5]) # +total time
                    self.arc_cost.append(float(dum[5])) # arc cost
                    self.arc_type.append(int(dum[1])) # arc type
                    if int(dum[1]) == 1:
                        # Boarding arc
                        self.line_boarding[int(dum[2])].append(int(dum[0]))

    #--------------------------------------------------------------------------
    def calculate(self, sol):
        """Calculates the constraint functions to determine feasibility.

        Accepts the current solution as an argument. This is then used to
        update the local Spiess model, which in turn is then used to calculate
        the network flows for use in calculating operator and user costs.

        Returns a tuple consisting of the feasibility result (1 or 0) followed
        by a list of the constraint function elements, which are:
            vehicle operation costs
            fare revenue
            total in-vehicle travel time
            total walking time
            total waiting time

        These constraint function elements are used within this method to
        evaluate the solution's feasibility, but the versions returned are the
        raw values free of coefficients. This is to make it easier to use the
        solution log to reevaluate feasibility if the coefficients ever change.
        """

        # Update line frequencies and capacities based on solution
        self._line_attributes(sol)

        # Use assignment model to find flow vector and waiting time
        # (note that the calculate() method returns a NumPy array of flows)
        flow, wait = self.spiess.calculate()

        # Use assignment output to calculate operator and user cost components
        (oc_operating, oc_fare, uc_travel, uc_walk,
         uc_wait) = self._cost_components(sol, list(flow), wait)

        # Use cost components to evaluate feasibility
        feas = self._feasibility(oc_operating, oc_fare, uc_travel, uc_walk,
                                 uc_wait)

        return feas, [oc_operating, oc_fare, uc_travel, uc_walk, uc_wait]

    #--------------------------------------------------------------------------
    def _cost_components(self, sol, flow, wait):
        """Calculates the components of the cost functions for a given flow.

        Requires the solution vector, the arc flow vector, and the total
        waiting time scalar.

        Calculates and returns a 5-tuple of cost components in the following
        order:
            vehicle operation cost (operator)
            money made from fares (operator)
            total in-vehicle travel time (user)
            total walking time (user)
            total waiting time (user)

        All of these components are returned as a positive value with no
        coefficient for use in the solution log.
        """

        # Calculate vehicle operating costs and fares
        oc_operating = 0 # operating cost
        oc_fare = 0 # fare revenue
        for i in range(len(sol)):
            oc_operating += self.vcost[self.vtype[i]]*sol[i] # cost for a line
            boards = 0 # total boards of the line
            for a in self.line_boarding[i]:
                boards += flow[a] # add boarding arc's flow
            oc_fare += self.vfare[self.vtype[i]]*boards # total line fares

        # Calculate user travel times
        uc_travel = 0 # total in-vehicle travel time
        uc_walk = 0 # total walking time
        for i in range(len(flow)):
            if self.arc_type[i] == 0:
                # Line arc
                uc_travel += flow[i]*self.arc_cost[i] # total time for arc
            if self.arc_type[i] == -1:
                # Walking arc
                uc_walk += flow[i]*self.arc_cost[i] # total time for arc

        return oc_operating, oc_fare, uc_travel, uc_walk, wait

    #--------------------------------------------------------------------------
    def _operator_cost(self, oc_operating, oc_fare):
        """Calculates operator cost given its two components.

        Requires the vehicle operating cost and fare revenue, respectively.
        """

        return self.time_horizon*oc_operating - oc_fare

    #--------------------------------------------------------------------------
    def _user_cost(self, uc_travel, uc_walk, uc_wait):
        """Calculates user cost given its three components.

        Requires the total in-vehicle travel time, walking time, and waiting
        time, respectively.
        """

        return (self.riding_weight*uc_travel + self.walking_weight*uc_walk
                + self.waiting_weight*uc_wait)

    #--------------------------------------------------------------------------
    def _feasibility(self, oc_operating, oc_fare, uc_travel, uc_walk, uc_wait):
        """Evaluates feasibility based on constraint function components.

        Requires five components corresponding to the outputs of the component
        calculation method (operator vehicle operation costs, operator fare
        revenue, user in-vehicle travel time, user walking time, user waiting
        time).

        The feasibility status is returned as an integer (0 for no, 1 for yes).
        """

        feasible = 1

        # Compare operator cost to (1+r) * initial cost
        if (self._operator_cost(oc_operating, oc_fare) >
            (1+self.operator_percent)*self.operator_init):
            feasible = 0

        # Compare user cost to (1+r) * initial cost
        if (self._user_cost(uc_travel, uc_walk, uc_wait) >
            (1+self.user_percent)*self.user_init):
            feasible = 0

        return feasible

    #--------------------------------------------------------------------------
    def _line_attributes(self, sol):
        """Calculates line frequencies and capacities for a given solution.

        Uses the solution vector to update the frequency and capacity of each
        line, and then updates the Spiess object with this new information.
        """

        # Update the line attributes
        for i in range(len(sol)):
            if sol[i] > 0:
                self.line_freq[i] = sol[i]/self.line_time[i]
                self.line_cap[i] = (self.vcapacity[self.vtype[i]]
                                    *self.line_freq[i])
            else:
                self.line_freq[i] = 0
                self.line_cap[i] = 0

        # Update the Spiess object transit attributes
        self.spiess.update_lines(self.line_freq, self.line_cap)

    #--------------------------------------------------------------------------
    def cost_calculation(self, sol):
        """Calculates the operator and user costs for a given solution.

        Useful for calculating the initial operator and user costs before
        setting up the rest of the problem. Uses the calculate() method, but
        instead of returning the full list of constraint components it returns
        only the cost function values.
        """

        _, [oc_operating, oc_fare, uc_travel, uc_walk,
            uc_wait] = self.calculate(sol)

        # Use cost components to calculate cost functions
        operator_cost = self._operator_cost(oc_operating, oc_fare)
        user_cost = self._user_cost(uc_travel, uc_walk, uc_wait)

        return operator_cost, user_cost

    #--------------------------------------------------------------------------
    def current_init(self, sol):
        """Resets initial costs based on the given solution.

        Calculates the operator and user costs for the given solution and uses
        them to overwrite this object's initial operator and user cost
        variables, effectively treating the given solution as the new initial
        solution.
        """

        self.operator_init, self.user_init = self.cost_calculation(sol)
