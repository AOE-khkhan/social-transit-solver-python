"""A module for reading and writing the current TS/SA memory structure.

At the end of the solution algorithm we write the memory structures to an
external file in order to allow a search to be picked up from where it left
off.
"""

#==============================================================================
class MemoryLog:
    """A class for reading and writing the memory log file.

    The object created from this class interacts with the memory log file. It
    includes methods for reading its contents and returning the memory
    structures, as well as for writing the current memory structures to the
    log file.

    The log file is arranged so that each row corresponds to a memory strucure
    in the following order:
        comment line explaining the row contents
        list of ADD tabu tenures
        list of DROP tabu tenures
        current solution
        best known solution
        current objective value
        best known objective value
        current iteration number
        inner nonimprovement counter
        outer nonimprovement counter
        current tabu tenure
        current SA temperature
        attractive objective list (single tab-separated row)
        attractive solution list (sequence of tab-separated rows)

    This is also the order of the arguments for the write method and the order
    of the tuple returned by the read method. Note that the attractive solution
    list can vary in length, which is why it is placed at the end of the log
    file.
    """

    #--------------------------------------------------------------------------
    def __init__(self, logfile="log/memory.txt"):
        """Memory log object constructor. Opens the memory log file.

        Accepts the following optional keyword arguments:
            logfile -- Path to the memory log file.
        """

        self.logfile = open(logfile, 'r+')

    #--------------------------------------------------------------------------
    def __del__(self):
        """Memory log destructor closes the open file."""

        self.logfile.close()

    #--------------------------------------------------------------------------
    def clear(self):
        """Clears the memory file."""

        self.logfile.seek(0)
        self.logfile.truncate(0)

    #--------------------------------------------------------------------------
    def load(self):
        """Reads the contents of the TS/SA memory structures from the log.

        The contents are returned as a tuple with the same order as the rows of
        the log file.
        """

        self.logfile.seek(0) # go to beginning of file
        self.logfile.readline() # skip comment line

        # Read in vectors
        tabu_add = [float(n) for n in self.logfile.readline().split()]
        tabu_drop = [float(n) for n in self.logfile.readline().split()]
        sol = [int(n) for n in self.logfile.readline().split()]
        sol_best = [int(n) for n in self.logfile.readline().split()]

        # Read in scalars
        obj = float(self.logfile.readline())
        obj_best = float(self.logfile.readline())
        iteration = int(self.logfile.readline())
        nonimp_in = int(self.logfile.readline())
        nonimp_out = int(self.logfile.readline())
        tenure = float(self.logfile.readline())
        tmp = float(self.logfile.readline())

        # Read in attractive solution objectives
        attractive_obj = [float(n) for n in self.logfile.readline().split()]

        # Read in attractive solutions
        attractive = []
        for line in self.logfile:
            attractive.append([int(n) for n in line.split()])

        return (tabu_add, tabu_drop, sol, sol_best, obj, obj_best, iteration,
                nonimp_in, nonimp_out, tenure, tmp, attractive_obj, attractive)

    #--------------------------------------------------------------------------
    def save(self, tabu_add, tabu_drop, sol, sol_best, obj, obj_best,
              iteration, nonimp_in, nonimp_out, tenure, tmp, attractive_obj,
              attractive):
        """Writes the contents of the TS/SA memory structures to the log.

        This clears the contents of the memory log file and replaces it with
        the given information. Its arguments have the same order as the rows of
        the log file.
        """

        self.logfile.seek(0)
        self.logfile.truncate(0) # clear log file

        # Write default comment line
        print("tabu_add, tabu_drop, sol, sol_best, obj, obj_best, nonimp_in,"
              +"nonimp_out, tenure, temperature, attractive_obj, attractive",
              file=self.logfile)

        # Tabu tenures
        line = ""
        for n in tabu_add:
            line += str(n)+"\t"
        print(line, file=self.logfile)
        line = ""
        for n in tabu_drop:
            line += str(n)+"\t"
        print(line, file=self.logfile)

        # Solution vectors
        line = ""
        for n in sol:
            line += str(n)+"\t"
        print(line, file=self.logfile)
        line = ""
        for n in sol_best:
            line += str(n)+"\t"
        print(line, file=self.logfile)

        # Scalars
        print(str(obj), file=self.logfile)
        print(str(obj_best), file=self.logfile)
        print(str(iteration), file=self.logfile)
        print(str(nonimp_in), file=self.logfile)
        print(str(nonimp_out), file=self.logfile)
        print(str(tenure), file=self.logfile)
        print(str(tmp), file=self.logfile)

        # Attractive solution objective list
        line = ""
        for n in attractive_obj:
            line += str(n)+"\t"
        print(line, file=self.logfile)

        # Attractive solution list
        for i in range(len(attractive)):
            line = ""
            for n in attractive[i]:
                line += str(n)+"\t"
            print(line, file=self.logfile)
