"""A module for searching and writing to the solution log files.

The solution log is a table of every previously-generated solution along with
its objective and constraint function values. Its primary purpose is to avoid
having to solve the assignment model for any given solution more than once.

Whenever a move is considered during the neighborhood search, we first check
the solution log to see whether it has already been considered. If so, we can
immediately look up the original results. If not, then its results are
generated and then added to the log.

We also have a final solution log which contains only the best known solution,
since this is meant to be the final output of the search algorithm.
"""

#==============================================================================
class SolutionLog:
    """A class for maintaining the solution log file.

    The object created from this class interacts with the solution log files.
    During the main driver's runtime, this class maintains the solution log in
    memory as a dictionary. The dictionary is created from the log file upon
    initialization, and is used to write to the log file upon deletion.

    The dictionary keys are simply string versions of a solution vector, and
    we have methods to convert a key to a vector and vice-versa. Each key is
    linked to a list corresponding to the columns of the log file. The columns
    are grouped in the following order:
        string of solution vector
        feasibility status
        list of constraint function elements
        time required to evaluate constraint function
        objective value
        secondary objective value
        time required to evaluate objective value

    The 'feasibility status' column should be either 1, 0, or -1. 1 indicates
    that the solution is definitely feasible, 0 indicates that it is definitely
    infeasible, and -1 is used to initialize placehoder rows whose feasibility
    has not yet been evaluated. This comes up during the neighborhood search
    portion of the main algorithm where we begin by sorting neighbors by
    objective value before evaluating their constraint functions.

    The 'objective value' and 'secondary objective value' are the minimum and
    sum of the 2SFCA metrics, respectively. The minimum is our real objective
    which the algorithm is actually attempting to optimize, while the secondary
    objective is only for use in breaking ties during the neighborhood search
    in the likely event that many neighbors produce the same minimum.

    The final solution file consists of only three rows: the solution vector,
    a vector of constraint function values, and the objective value,
    respectively.
    """

    #--------------------------------------------------------------------------
    def __init__(self, logfile="log/solution.txt", solfile="log/final.txt",
                 sol_length=10, con_length=1, pickup=True):
        """Opens solution log file and initializes solution dictionary.

        Accepts the following optional keyword arguments:
            logfile -- Path to the solution log file.
            solfile -- Path to the final solution file.
            sol_length -- Number of solution vector elements.
            con_length -- Number of constraint vector elements.
            pickup -- Whether to initialize the solution dictionary by reading
                the solution log. Defaults to True.
        """

        self.logfile = logfile
        self.solfile = solfile
        self.sol_length = sol_length
        self.con_length = con_length
        self.sol_dic = {}
        if pickup == True:
            self._load()

    #--------------------------------------------------------------------------
    def __del__(self):
        """Saves dictionary to solution log upon deletion."""

        self._save()
        del self.sol_dic

    #--------------------------------------------------------------------------
    def _load(self):
        """Reads contents of solution log into solution dictionary."""

        with open(self.logfile, 'r') as f:
            f.readline() # skip comment line
            for line in f:
                lst = line.split()

                # Format each column of the log file
                feas = [int(lst[1])]
                con = [float(n) for n in lst[2:2+self.con_length]]
                con_time = [float(lst[2+self.con_length])]
                obj = [float(lst[3+self.con_length])]
                obj2 = [float(lst[4+self.con_length])]
                obj_time = [float(lst[5+self.con_length])]

                # Combine information into a list in the dictionary
                self.sol_dic[lst[0]] = (feas + con + con_time + obj + obj2 +
                            obj_time)

    #--------------------------------------------------------------------------
    def _save(self):
        """Writes contents of solution dictionary to solution log."""

        with open(self.logfile, 'w') as f:
            # Write comment line
            line = "sol\tfeas\t"
            for i in range(self.con_length):
                line += "con["+str(i)+"]\t"
            line += "con_time\tobj_min\tobj_sum\tobj_time\t"
            print(line, file=f)

            # Write dictionary to file
            for sol in self.sol_dic:
                # "sol" here is the key string for the given solution
                line = sol+"\t"
                for n in self.sol_dic[sol]:
                    line += str(n)+"\t"
                print(line, file=f)

    #--------------------------------------------------------------------------
    def _sol2key(self, sol):
        """Converts a solution vector to a key string."""

        key = ""
        for elem in sol:
            key += str(elem)+"&"

        return key[:-1] # skip final blank after "&"

    #--------------------------------------------------------------------------
    def _key2sol(self, key):
        """Converts a key string to a solution vector."""

        sol = [0 for i in range(self.sol_length)]
        lst = key.split("&")
        for i in range(len(lst)):
            sol[i] = int(lst[i])

        return sol

    #--------------------------------------------------------------------------
    def search(self, sol):
        """Looks for a given solution vector in the solution dictionary.

        Returns a boolean indicating whether the solution has been logged. This
        includes all solutions that have a key in the dictionary, some of which
        may only be placeholders, which is indicated by a feasibility status
        value of -1.
        """

        if self._sol2key(sol) in self.sol_dic:
            # True if the key version of the given solution is in sol_dic
            return True
        else:
            return False

    #--------------------------------------------------------------------------
    def lookup(self, sol):
        """Returns the solution information from a given solution.

        The information is returned as a triple consisting of the feasibility
        status, the constraint vector, and the objective value, respectively.
        """

        lst = self.sol_dic[self._sol2key(sol)] # raw dictionary entry

        # Separate raw list into the necessary pieces
        feas = lst[0]
        con = lst[1:1+self.con_length]
        obj = lst[2+self.con_length]
        obj2 = lst[3+self.con_length]

        return feas, con, obj, obj2

    #--------------------------------------------------------------------------
    def create(self, sol, feas, con, con_time, obj, obj2, obj_time):
        """Creates a new dictionary item for a given solution.

        The required arguments are, in order:
            solution vector
            feasibility status
            list of constraint function elements
            time required to evaluate constraint function
            objective value
            secondary objective value
            time required to evaluate objective value

        If the specified solution is already present in the solution
        dictionary, this will update its information. If not, a new entry will
        be created for it.
        """

        self.sol_dic[self._sol2key(sol)] = ([feas] + con + [con_time] + [obj] +
                     [obj2] + [obj_time])

    #--------------------------------------------------------------------------
    def update(self, sol, feas=None, con=None, con_time=None, obj=None,
               obj2=None, obj_time=None):
        """Updates the information for a given dictionary item.

        Requires a solution vector, which must already be logged. Includes
        keyword arguments for each individual element of the dictionary entry.
        Unspecified keyword arguments will be left unchanged.

        Accepts the following keyword arguments to set individual elements:
            feas -- Feasibility status.
            con -- Constraint vector.
            con_time -- Constraint calculation time.
            obj -- Objective value.
            obj2 -- Secondary objective value.
            obj_time -- Objective calculation time.
        """

        key = self._sol2key(sol)

        # Update feasibility status
        if feas != None:
            self.sol_dic[key][0] = feas

        # Update constraint vector
        if con != None:
            self.sol_dic[key][1:1+self.con_length] = con[:]

        # Update constraint time
        if con_time != None:
            self.sol_dic[key][1+self.con_length] = con_time

        # Update objective value
        if obj != None:
            self.sol_dic[key][2+self.con_length] = obj

        # Update secondary objective value
        if obj2 != None:
            self.sol_dic[key][3+self.con_length] = obj2

        # Update objective time
        if obj_time != None:
            self.sol_dic[key][4+self.con_length] = obj_time

    #--------------------------------------------------------------------------
    def final(self, sol, obj):
        """Writes to the final solution file."""

        with open(self.solfile, 'w') as f:
            # Write solution vector
            line = ""
            for elem in sol:
                line += str(elem)+"\t"
            print(line, file=f)

            print(str(obj), file=f) # write objective
