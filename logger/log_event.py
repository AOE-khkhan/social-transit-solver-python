"""A module for writing to the event log files.

We maintain two event logs: a descriptive event log to catalog the progress of
the search algorithm while it runs, and a summarized log file that just
includes the current and best objective values after each iteration.

The event log is mostly meant to aid in the process of troubleshooting and
parameter tuning, while the objective log is meant to aid in the process of
quickly generating plots of the objective value over time.
"""

import numpy as np

#==============================================================================
class EventLog:
    """A class for logging the solution algorithm progress.

    The object created from this class is equipped to write to specified files.
    Its methods correspond to various types of event and write to the log file
    according to a template.

    Creating a log object also creates and opens its log files. These files
    remain open until the object is deleted.
    """

    #--------------------------------------------------------------------------
    def __init__(self, logfile="log/event.txt", objfile="log/objective.txt",
                 pickup=False):
        """Event log object constructor. Creates and opens log files.

        Be aware that this will overwrite existing files with the same name.

        Accepts the following optional keyword arguments:
            logfile -- Path to the event log file.
            objfile -- Path to the objective value output.
            pickup -- Indicates whether we are picking up from a previous
                search. Defaults to False. If True, we append to the existing
                logs, while if False, we start with clear log files.
        """

        if pickup == True:
            self.logfile = open(logfile, 'a')
            print("\n"+("#"*100), file=self.logfile)
            print("Search Restart", file=self.logfile)
            print(("#"*100)+"\n", file=self.logfile)
            self.objfile = open(objfile, 'a')
        else:
            self.logfile = open(logfile, 'w')
            self.objfile = open(objfile, 'w')

    #--------------------------------------------------------------------------
    def __del__(self):
        """Event log destructor closes all open files."""

        self.logfile.close()
        self.objfile.close()

    #--------------------------------------------------------------------------
    def event_clear(self):
        """Clears the event log."""

        self.logfile.seek(0)
        self.logfile.truncate(0)

    #--------------------------------------------------------------------------
    def write(self, string):
        """Writes a given string to the event log file."""

        print(string, file=self.logfile)

    #--------------------------------------------------------------------------
    def intro(self, sol, obj):
        """Logs the initial solution and objective."""

        string = "Initial solution:\n"
        for n in sol:
            string += str(n)+"\t"
        string += "\nInitial objective: "+str(obj)
        print(string, file=self.logfile)

    #--------------------------------------------------------------------------
    def iteration_header(self, it, it_max):
        """Writes a header for the current iteration."""

        print("\n"+("="*80), file=self.logfile)
        print("Iteration "+str(it+1)+" / "+str(it_max), file=self.logfile)
        print(("="*80)+"\n", file=self.logfile)

    #--------------------------------------------------------------------------
    def exhaustive_header(self):
        """Writes a header for the final exhaustive search."""

        print("\n"+("="*80), file=self.logfile)
        print("Final exhaustive search", file=self.logfile)
        print(("="*80)+"\n", file=self.logfile)

    #--------------------------------------------------------------------------
    def nbhd_list(self, movetype, lst):
        """Writes a list of neighbors of a given move type."""

        string = movetype+" neighborhood:\n"
        for elem in lst:
            string += str(elem)+"\t"
        print(string, file=self.logfile)

    #--------------------------------------------------------------------------
    def aspiration(self, movetype, move):
        """Reports a move whose tabu status is ignored due to improved best."""

        print("Tentatively keeping tabu "+movetype+" move "+str(move)+" due to"
              +" improved best.", file=self.logfile)

    #--------------------------------------------------------------------------
    def nbhd_stats(self, lookups, new, tot_time):
        """Writes a report for some neighborhood search statistics."""

        print("\nLooked up "+str(lookups)+" solutions.",
              file=self.logfile)
        print("Calculated "+str(new)+" solutions.", file=self.logfile)
        print("Total search time: "+str(tot_time), file=self.logfile)

    #--------------------------------------------------------------------------
    def best_moves(self, move1, obj1, move2, obj2):
        """Summarizes the chosen pair of best moves."""

        print("\n1st best move "+self._move_format(move1)+" with value "
              +str(obj1), file=self.logfile)
        if obj2 < np.inf:
            print("2nd best move "+self._move_format(move2)+" with value "
                  +str(obj2), file=self.logfile)
        else:
            print("No 2nd best move.", file=self.logfile)

    #--------------------------------------------------------------------------
    def improvement(self, move, decrease):
        """Summarizes the results of an improvement iteration."""

        print("\nImprovement iteration!", file=self.logfile)
        print("Making move "+self._move_format(move), file=self.logfile)
        print("Objective decreased by "+str(decrease), file=self.logfile)

    #--------------------------------------------------------------------------
    def nonimprovement_pass(self, prob, move1, move2):
        """Summarizes a nonimprovement iter that passes the SA criterion."""

        print("\nNonimprovement iteration.\nPassed SA criterion with "
              +"probability "+str(prob), file=self.logfile)
        print("Making move "+self._move_format(move1), file=self.logfile)
        if (move2[0] > 0) and (move2[1] > 0):
            print("Keeping attractive move "+self._move_format(move2),
                  file=self.logfile)

    #--------------------------------------------------------------------------
    def nonimprovement_fail(self, prob, move):
        """Summarizes a nonimprovement iter that fails the SA criterion."""

        print("\nNonimprovement iteration.\nFailed SA criterion with "
              +"probability "+str(prob), file=self.logfile)
        print("Keeping attractive move "+self._move_format(move),
              file=self.logfile)

    #--------------------------------------------------------------------------
    def local_move(self, move, obj):
        """Reports a given move and new objective value."""

        print("Making local move "+self._move_format(move)+" with value "
              +str(obj), file=self.logfile)

    #--------------------------------------------------------------------------
    def new_best(self, obj):
        """Reports an update to the best known objective."""

        print("New best objective! "+str(obj), file=self.logfile)

    #--------------------------------------------------------------------------
    def nonimp_in(self, tenure, sol):
        """Summarizes the results of the inner counter maxing out."""

        print("\nInner nonimprovement counter maxed out.", file=self.logfile)
        print("Tenures increasing to"+str(tenure), file=self.logfile)
        string = "Jumping to the following attractive solution:\n"
        for n in sol:
            string += str(n)+"\t"
        print(string, file=self.logfile)

    #--------------------------------------------------------------------------
    def nonimp_out(self, tenure):
        """Summarizes the results of the outer counter maxing out."""

        print("\nOuter nonimprovement counter maxed out. Resetting tenures to "
              +str(tenure), file=self.logfile)

    #--------------------------------------------------------------------------
    def _move_format(self, move):
        """Creates a string to more naturally describe a move."""

        if move[1] < 0:
            return "ADD("+str(move[0])+")"
        elif move[0] < 0:
            return "DROP("+str(move[1])+")"
        else:
            return "SWAP("+str(move[0])+"<<"+str(move[1])+")"

    #--------------------------------------------------------------------------
    def obj_clear(self):
        """Clears the objective log and rewrites the comment line."""

        self.objfile.seek(0)
        self.objfile.truncate(0)
        print("iter\tcurrent\tbest", file=self.objfile)

    #--------------------------------------------------------------------------
    def final(self, sol, obj):
        """Writes the final solution and objective."""

        print("\n"+("="*80), file=self.logfile)
        print("Final results", file=self.logfile)
        print(("="*80)+"\n", file=self.logfile)

        print("Solution:\n", sol, file=self.logfile)
        print("\nObjective:", obj, file=self.logfile)

    #--------------------------------------------------------------------------
    def obj_append(self, iteration, obj, obj_best):
        """Adds a new line to the objective log.

        The arguments are the iteration number, current objective, and best
        objective, respectively.
        """

        print(str(iteration)+"\t"+str(obj)+"\t"+str(obj_best),
              file=self.objfile)
