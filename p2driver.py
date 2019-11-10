"""The tabu search/simulated annealing hybrid algorithm.

This will act as the main driver for solving the project 2 problem. It should
be run as a script, and will import modules to conduct the major important
steps of the algorithm.

The TS/SA algorithm relies on a large number of tuning parameters. These
parameters should remain constant while the algorithm is running. A variety of
log files are maintained to store solution data and memory structures so that
the search may be halted and resumed.
"""

# Next changes:
#   Test current version with CapCon and then Mandl networks.
#   Switch tiebreaker objective to gravity metric of current lowest 2SFCA metric.
#   Change fare revenue to only depend on the total number of passengers.
#   Try tuning parameters with Mandl network to see that things are moving along.
#   Process real data, test, and possibly work on a standalone C++ implementation.
#   Try to get a few iterations (~100) of each trial set done by the end of the
#       month. Write programs to process the data automatically and write up
#       the results in the thesis, so that this project can be considered
#       technically complete by the beginning of the semester. During the
#       semester we can be generating more data and using the programs to
#       automatically update the data sets in the paper.

# Advice for parameter tuning:
# A lot of parameters essentially allow us to choose whether to focus more on
#   intensification or diversification. Run some tests to see which appears to
#   be more important. Try to develop metrics for when we appear to be getting
#   trapped in a limited area, or when we seem to be wandering for too long
#   without an objective improvement.
# Choose neighborhood sizes based on how long the neighborhood search is
#   taking. Larger neighborhoods means a more exhaustive search and therefore
#   more intensification, but might also be more computationally expensive.
#   Later on we might even dynamically adjust the neighborhood size to
#   correspond to intensification/diversification phases.
# Tabu tenures and increments control how much we focus on intensification
#   versus diversification.
# Initial temperature and cooling schedule controls how early in the proceess
#   we begin tightening our restrictions, which essentially corresponds to
#   a general move towards intensification.
# The inner/outer counter cutoffs give us some more explicit control over when
#   to switch between intensification and diversification.

# Run some tests to see relatively how long it takes to calculate the
# constraint function values versus the objective value. This program was
# written to try to minimize the required number of constraint function
# evaluations, but at the cost of running a bunch of preliminary objective
# function evaluations. If it turns out that those are also expensive, we may
# need to rethink the structure of the neighborhood search.

# If the constraint calculations are taking way too long and it is becoming a
# huge issue, then consider versions of the program that involve fewer
# feasibility evaluations, including the following:
#   Start by taking bigger steps, which moves us around the solutions pace
#       faster, and let the step size trail off as the algorithm moves forward.
#       This could also be tied to diversification and intensification.
#   It may be necessary to code a C++ version of this entire program, but that
#       should only be undertaken once everything else is already in place and
#       running.

# If the search seems to be wandering a lot due to the lack of gradient in the
# objective, try using the gravity objective as a secondary objective instead
# of the sum of 2SFCA metrics. It seems likely that only an extreme change from
# the current solution would have a chance of increasing the minimum, but the
# algorihm might get caught in an extremely flat spot where it has no idea
# which direction to head in. The gravity metric is continuous, so all local
# changes should affect it. Either go with the min or sum of gravity metrics.

# The more I think about it, the more I am unconvinced that this project is
# justifiable. Setting aside the mismatch between fleet assignments for rush
# hour traffic and for infrequent primary care-related traffic, the way our
# model is currently set up attempts to maximize peoples' access to facilities.
# That access measure ultimately depends on pairwise travel times, and the only
# way that that depends on the fleet sizes is waiting times, which are likely
# an insignificant part of the overall trip length unless there is a massive
# change, and in that case capacity will be much more of an issue than time.
# I simply do not see someone basing their decision for whether to seek medical
# care as depending on how long they would have to wait for the bus. With a
# different social access goal, then maybe, but certainly not health care. We
# could either try to sell this as being purely a proof of concept, since the
# 2SFCA metric is a reasonable thing to use for any type of access, or we could
# work on something more drastic.
#
# In the next progress report, write to Dr. K. to float these ideas:
#
# For this reason it might be better to try considering the route design
# problem again. The easiest implementation would include a finite set of
# specific routes, but more complicated versions could have us constructing a
# route link-by-link with constraints to enforce that it makes sense as a
# route.
# All of the tentative routes would be included in the route set just like all
# the rest, but we would also have a binary decision (controlled within the
# TS/SA algorithm somehow) that would indicate which routes exist, and enforce
# constraints that the non-existant routes have a fleet size of zero. If we
# make a decision to switch one tentative route for another, we also transfer
# their entire fleet. This could lead to continuity issues, since these route
# existence decisions have way more impact than the fleet size decisions.
# Alternatively, say we just want to add one new route, and we've already
# decided on how many buses to add (they're allowed to come from nowhere,
# assuming that CTA is buying new buses). Then we attempt to solve the problem
# of designing this new route assuming that all fleet sizes are fixed. The
# decisions now are which links to include in the new route. It could start
# with a fixed structure based on a heuristically good choice, or it could just
# start as a single link somewhere, and our local moves would consist of either
# extending one endpoint, removing one endpoint, or rewiring one endpoint, with
# constraints to make sure we don't get any crossings or anything.
# As a simpler alternative to this, we could instead heuristically pick some
# new route locations likely to be helpful (read some papers about how solution
# pools for route designs are generated; something to do with gravity). Then
# simply try re-solving the model with the addition of each of these new
# routes. This is similar to the thing we wanted to try with adding the new
# CTA projects (Red Line extension, Orange Line extension, Ashand rapid bus),
# except that these new projects would be specifically designed to address the
# access issue. This could still be sold as a relatively minor thing that CTA
# could do to drastically improve access.
# As for how to heuristically generate good routes, it should probably have
# something to do with trying to connect areas with particularly low access to
# areas with particularly high access. Express routes are also an easy option.
#
# Run some tests to see how sensitive everything appears to be to the number of
# buses on a route. I would suspect that it hardly matters except for the
# difference between 0 and 1. If that is the case, then we don't need to pay
# much attention to specific fleet sizes for the purposes of the objective.
#
# Note that most of this would probably take all the time I have left this
# fall, and would probably have to happen instead of, not in addition to, the
# multiobjective study where we try varying the allowed percentage increase.

import objective.obj_2sfca as ob
import constraints.constraints as con
import logger.log_event as elog
import logger.log_solution as slog
import logger.log_memory as mlog
import numpy as np
from time import time
import operator as op

#==============================================================================
# Main Driver
#==============================================================================

def driver(max_iter=50, exhaustive=False, swaps=True, pickup=False,
           display=False, percent=0.0):
    """The main driver of the TS/SA hybrid algorithm.

    Accepts the following optional keyword arguments:
        max_iter -- Number of iterations for the TS/SA hybrid algorithm.
            Defaults to 50.
        exhaustive -- Whether or not to conduct an exhaustive local search
            after reaching the iteration cutoff. Defaults to False.
        swaps -- Whether or not to consider SWAP moves in the exhaustive
            search. Defaults to True. Set to False to limit the size of the
            neighborhood in case the final search is taking too long.
        pickup -- Whether or not to begin by loading the memory structures of
            the previous search. Defaults to False, in which case the memory
            log is cleared and we start from scratch.
        display -- Whether or not to display the progress of the algorithm to
            the command line. Defaults to False.
        percent -- Allowable increase in the operator and user cost functions.
            Defaults to 0.0. This essentially defines the feasibility
            constraints, and one of the most important goals of this study is
            to try solving the model for different values of this percentage.
    """

    global sol, obj, sol_best, obj_best, nonimp_in, nonimp_out, tabu_add
    global tabu_drop, tenure, tmp, attractive_obj, attractive, vcurrent
    global Log, SolLog, MemLog, Obj, Con

    #--------------------------------------------------------------------------
    # Initialization
    #--------------------------------------------------------------------------

    if display == True:
        print("Initialization")

    _parameter_load() # load search parameters from file
    _data_load() # load problem data from file

    # Initialize constraint and objective objects
    Con = con.Constraint(data="data/", operator_percent=percent,
                         user_percent=percent)
    Obj = ob.Objective(data="data/", logfile="log/metrics.txt")

    # Initialize variables
    sol = sol_init[:] # current solution
    sol_best = sol[:] # best known solution
    obj, _ = Obj.calculate(sol) # current objective value
    obj_best = obj # best known objective value
    tmp = tmp_init # SA temperature
    tenure = tenure_init # tabu tenure
    nonimp_in = 0 # inner nonimprovement counter (triggers diversification)
    nonimp_out = 0 # outer nonimprovement counter (triggers intensification)
    vcurrent = [0 for i in range(len(vbound))] # tot num of each vehicle type
    for i in range(len(sol)):
        vcurrent[vtype[i]] += sol[i]
    tabu_add = [0 for i in range(len(sol))] # ADD tabu tenures (0 if non-tabu)
    tabu_drop = [0 for i in range(len(sol))] # DROP tabu tenures
    attractive = [] # attractive solution list
    attractive_obj = [] # attractive solution objectives
    iteration = 0 # iteration number for use in cooling schedule

    # Initialize logger objects
    Log = elog.EventLog(logfile="log/event.txt", objfile="log/objective.txt",
                        pickup=pickup)
    SolLog = slog.SolutionLog(logfile="log/solution.txt",
                              solfile="log/final.txt", sol_length=len(sol),
                              con_length=con_length, pickup=pickup)
    MemLog = mlog.MemoryLog(logfile="log/memory.txt")

    if pickup == True:
        # Read the logged memory structures into the variables
        (tabu_add, tabu_drop, sol, sol_best, obj, obj_best,
         iteration, nonimp_in, nonimp_out, tenure, tmp,
         attractive_obj, attractive) = MemLog.load()
        vcurrent = [0 for i in range(len(vbound))] # update vehicle numbers
        for i in range(len(sol)):
            vcurrent[vtype[i]] += sol[i]
    else:
        # Clear the memory and reset the logs
        MemLog.clear()
        Log.event_clear()
        Log.obj_clear()
        Con.current_init(sol) # base operator/user costs on initial solution

    #--------------------------------------------------------------------------
    # Main loop
    #--------------------------------------------------------------------------

    Log.intro(sol, obj)

    for k in range(max_iter):

        Log.iteration_header(k, max_iter)
        if display == True:
            print("Iteration", k+1, "/", max_iter)

        # Neighborhood search for two best moves
        b1, b2 = _neighborhood_search()
        sol1 = b1[0:2]
        obj1 = b1[2]
        type1 = b1[3]
        sol2 = b2[0:2]
        obj2 = b2[2]

        Log.best_moves(sol1, obj1, sol2, obj2)

        #----------------------------------------------------------------------
        # Improvement iteration
        if obj1 < obj:

            Log.improvement(sol1, obj-obj1)

            nonimp_out = 0 # reset outer nonimprovement counter
            tenure = tenure_init # reset tabu tenures
            sol = _sol_move(add=sol1[0], drop=sol1[1]) # update solution
            obj = obj1 # update current objective

            # Update current vehicles in use
            if sol1[0] >= 0:
                vcurrent[type1] += 1
            if sol1[1] >= 0:
                vcurrent[type1] -= 1

            # Make undoing the current move tabu
            if sol1[0] >= 0:
                tabu_drop[sol1[0]] = tenure
            if sol1[1] >= 0:
                tabu_add[sol1[1]] = tenure

            # Update best solution if needed
            if obj < obj_best:
                sol_best = sol[:]
                obj_best = obj
                Log.new_best(obj_best)

        #----------------------------------------------------------------------
        # Nonimprovement iteration
        else:

            nonimp_out += 1 # increment both counters
            nonimp_in += 1

            # SA criterion
            if np.random.rand() < np.exp(-(obj1-obj)/tmp):

                _tenure_increment() # increase tenures
                sol = _sol_move(add=sol1[0], drop=sol1[1]) # update solution

                # Update current vehicles in use
                if sol1[0] >= 0:
                    vcurrent[type1] += 1
                if sol1[1] >= 0:
                    vcurrent[type1] -= 1

                # Make undoing the current move tabu
                if sol1[0] >= 0:
                    tabu_drop[sol1[0]] = tenure
                if sol1[1] >= 0:
                    tabu_add[sol1[1]] = tenure

                nonimp_in = 0 # reset inner counter

                # Keep the second best solution as attractive
                if obj2 < np.inf:
                    # Skip in the rare event that sol2 does not exist
                    attractive.append(_sol_move(add=sol2[0], drop=sol2[1]))
                    attractive_obj.append(obj2)

                Log.nonimprovement_pass(np.exp(-(obj1-obj)/tmp), sol1, sol2)

            else:

                # Keep the best solution as attractive
                attractive.append(_sol_move(add=sol1[0], drop=sol1[1]))
                attractive_obj.append(obj1)

                Log.nonimprovement_fail(np.exp(-(obj1-obj)/tmp), sol1)

        #----------------------------------------------------------------------
        # End-of-iteration updates

        if len(attractive) > attractive_max:
            # Delete a random attractive solution if the list is too long
            i = np.random.randint(len(attractive))
            del attractive[i]
            del attractive_obj[i]

        if nonimp_in >= nonimp_in_max:
            # Inner counter cutoff indicates need for diversification
            nonimp_in = 0
            nonimp_out += 1
            _tenure_increment()

            # Replace current solution with a random attractive solution
            i = np.random.randint(len(attractive))
            sol = attractive[i][:]
            obj = attractive_obj[i]
            del attractive[i]
            del attractive_obj[i]

            # Recalculate current vehicles in use
            vcurrent = [0 for i in range(len(vbound))]
            for i in range(len(sol)):
                vcurrent[vtype[i]] += sol[i]

            Log.nonimp_in(tenure, sol)

        if nonimp_out >= nonimp_out_max:
            # Outer counter cutoff indicates need for intensification
            tenure = tenure_init
            Log.nonimp_out(tenure)

        # Update tabu tenures
        for i in range(len(tabu_add)):
            tabu_add[i] = max(tabu_add[i]-1, 0)
        for i in range(len(tabu_drop)):
            tabu_drop[i] = max(tabu_drop[i]-1, 0)

        # Update temperature according to cooling schedule
        iteration += 1
        _cooling(iteration)

        # Log the current objective value
        Log.obj_append(iteration, obj, obj_best)

    #--------------------------------------------------------------------------
    # End of main loop
    #--------------------------------------------------------------------------

    if exhaustive == True:
        # Exhaustive local search starting from best known solution
        if display == True:
            print("Final exhaustive search")
        _exhaustive_search(sol_best, obj_best, swaps=swaps)

    # Log memory structures
    MemLog.save(tabu_add, tabu_drop, sol, sol_best, obj, obj_best, iteration,
                  nonimp_in, nonimp_out, tenure, tmp, attractive_obj,
                  attractive)

    # Output the best known solution
    SolLog.final(sol_best, obj_best)
    Log.final(sol_best, obj_best)
    Obj.output(sol_best)

    # End logging
    del Log
    del SolLog
    del MemLog

    if display == True:
        print("All done!")

#==============================================================================
# Subroutines
#==============================================================================

def _parameter_load():
    """Loads TS/SA search parameters from the input file."""

    global tmp_init, tmp_factor, attractive_max, nbhd_add_lim, nbhd_add_lim2
    global nbhd_drop_lim, nbhd_drop_lim2, nbhd_swap_lim, tenure_init
    global tenure_factor, nonimp_in_max, nonimp_out_max, step

    # Read in parameters
    with open("data/Search_Parameters.txt", 'r') as f:
        f.readline() # skip comment line
        tmp_init = float(f.readline()) # initial SA temperature
        tmp_factor = float(f.readline()) # temperature cooling factor
        attractive_max = int(f.readline()) # max attractive solutions to store
        nbhd_add_lim = int(f.readline()) # 1st pass ADD neighborhood size
        nbhd_add_lim2 = int(f.readline()) # 2nd pass (final) ADD nbhd size
        nbhd_drop_lim = int(f.readline()) # 1st pass DROP neighborhood size
        nbhd_drop_lim2 = int(f.readline()) # 2nd pass DROP neighborhood size
        nbhd_swap_lim = int(f.readline()) # SWAP neighborhood size
        tenure_init = float(f.readline()) # initial tabu tenures
        tenure_factor = float(f.readline()) # tenure increase factor
        nonimp_in_max = int(f.readline()) # inner nonimp counter cutoff
        nonimp_out_max = int(f.readline()) # outer nonimp counter cutoff
        step = int(f.readline()) # step size of local moves

#------------------------------------------------------------------------------
def _data_load():
    """Loads initial solution data from the input file."""

    global vbound, vcapacity, con_length
    global lb, ub, sol_init, vtype

    # Read in problem data
    with open("data/Problem_Data.txt", 'r') as f:
        f.readline() # comment line
        vbound = [int(n) for n in f.readline().split()] # vehicle type limits
        vcapacity = [int(n) for n in f.readline().split()] # vehicle seat cap
        for i in range(4):
            # Skip vcost, vfare, operator_init, user_init
            f.readline()
        con_length = int(f.readline()) # constraint vector length

    # Read in transit data
    lb = [] # line fleet lower bounds
    ub = [] # line fleet upper bounds
    sol_init = [] # initial fleet sizes
    vtype = [] # fleet vehicle types
    i = -1
    with open("data/Transitdata.txt", 'r') as f:
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                lb.append(int(dum[3]))
                ub.append(int(dum[4]))
                sol_init.append(int(dum[6]))
                vtype.append(int(dum[7]))

#------------------------------------------------------------------------------
def _neighborhood_search():
    """Searches a random subset of neighbors and returns the two best.

    Considers random subsets of the possible ADD/DROP/SWAP moves until either
    reaching the cutoff specified by the search parameters or exhausting its
    possibilities. All solutions considered must obey bound feasibility
    constraints, feasibility function constraints, and tabu rules. If too few
    candidates are found, tabu rules are deleted. Tabu rules can also be broken
    for solutions that improve the best known objective.

    Returns a tuple of 4-element lists. The first list corresponds to the best
    candidate while the second list corresponds to the second best. Each list
    takes the form [add, drop, objective, vtype], where 'add' and 'drop' are
    the indices of the ADD/DROP moves (-1 if no change), 'objective' is that
    solution's objective value, and 'vtype' is the vehicle type involved in the
    move.
    """

    global nonimp_out, Log, step

    start = time() # begin neighborhood search timer

    nbhd_add = [] # final list of ADD neighbors
    obj_add = [] # final list of ADD neighbor objectives
    cand_add = [i for i in range(len(sol))] # candidate elements for ADDing
    nbhd_drop = [] # final list of DROP neighbors
    obj_drop = [] # final list of DROP neighbor objectives
    cand_drop = [i for i in range(len(sol))] # candidate elements for DROPping
    nbhd_swap = [] # final list of SWAP neighbors
    obj_swap = [] # final list of SWAP neighbor objectives

    repeat = True # indicates whether the ADD/DROP search must be repeated
    lu = 0 # tally of successful lookups
    nw = 0 # tally of newly-generated solutions

    while repeat == True:

        repeat = False # will be set back to True if we find no candidates

        #----------------------------------------------------------------------
        # ADD move selection

        # We will use a list of lists for storing tentative moves and their
        # information. Each row corresponds to a tentative move, while the
        # columns (in order) are: the ADD move, the move's objective, the
        # move's tiebreaker objective, and a boolean of whether the move has
        # already been added to the solution log.
        # These are being stored in a single array to aid in the sorting
        # process, since corresponding rows must be sorted into the same
        # position.
        tent = []

        # Tentative neighborhood selection (ignores constraints)
        while (len(tent) < nbhd_add_lim) and (len(cand_add) > 0):
            # Consider more ADD moves until reaching our limit or running out
            choice = np.random.choice(cand_add) # select a random candidate
            cand_add.remove(choice) # remove the candidate as a possibility
            if (sol[choice] + step) > ub[choice]:
                # Don't attempt an ADD that would exceed an upper bound
                continue
            if vcurrent[vtype[choice]] + step > vcapacity[vtype[choice]]:
                # Don't attempt an ADD that would exceed the vehicle limit
                continue
            sol_cand = _sol_move(add=choice) # generate candidate sol vector
            # Look up solution for feasability status and objective values
            cand_feas, _, cand_obj, cand_obj2 = _obj_lookup(sol_cand)
            if cand_feas == 0:
                # Skip moves known to be infeasible
                lu += 1
                continue
            if tabu_add[choice] > 0:
                if cand_obj >= obj_best:
                    # Skip tabu moves unless they improve our best objective
                    continue
                Log.aspiration("ADD", choice)
            # Flag new candidates
            if cand_feas < 0:
                tent.append([choice, cand_obj, cand_obj2, True])
            else:
                tent.append([choice, cand_obj, cand_obj2, False])

        # Sort the candidates lexicographically in ascending order of objective
        # value and then ascending order of secondary objective value.
        tent.sort(key = op.itemgetter(1, 2))

        # Store the results using more convenient list names
        nbhd_tent = [row[0] for row in tent]
        obj_tent = [row[1] for row in tent]
        new_tent = [row[3] for row in tent]

        for i in range(len(nbhd_tent)):

            # Consider tentative ADD moves to add to final list
            sol_cand = _sol_move(add=nbhd_tent[i])

            # Process new solutions
            if new_tent[i] == True:
                # Calculate constraints for new solutions
                cand_feas = _con_update(sol_cand, obj_tent[i])
                nw += 1
            else:
                # Otherwise the constraint is definitely feasible
                cand_feas = 1
                lu += 1

            if cand_feas > 0:
                # Add feasible solutions to neighborhood
                nbhd_add.append(nbhd_tent[i])
                obj_add.append(obj_tent[i])
            else:
                # Skip infeasible solutions
                continue

            if len(nbhd_add) >= nbhd_add_lim2:
                # Break if we've found enough ADD neighbors
                break

        Log.nbhd_list("ADD", nbhd_add)

        #----------------------------------------------------------------------
        # DROP move selection

        # We reuse the same tentative move array
        tent = []

        # Tentative neighborhood selection (ignores constraints)
        while (len(tent) < nbhd_drop_lim) and (len(cand_drop) > 0):
            # Consider more DROP moves until reaching our limit or running out
            choice = np.random.choice(cand_drop) # select a random candidate
            cand_drop.remove(choice) # remove the candidate as a possibility
            if (sol[choice] - step) < lb[choice]:
                # Don't attempt a DROP that would fall below a lower bound
                continue
            sol_cand = _sol_move(drop=choice) # generate candidate sol vector
            # Look up solution for feasability status and objective values
            cand_feas, _, cand_obj, cand_obj2 = _obj_lookup(sol_cand)
            if cand_feas == 0:
                # Skip moves known to be infeasible
                lu += 1
                continue
            if tabu_drop[choice] > 0:
                if cand_obj >= obj_best:
                    # Skip tabu moves unless they improve our best objective
                    continue
                Log.aspiration("DROP", choice)
            # Flag new candidates
            if cand_feas < 0:
                tent.append([choice, cand_obj, cand_obj2, True])
            else:
                tent.append([choice, cand_obj, cand_obj2, False])

        # Sort the candidates lexicographically as before
        tent.sort(key = op.itemgetter(1, 2))

        # Store the results using more convenient list names
        nbhd_tent = [row[0] for row in tent]
        obj_tent = [row[1] for row in tent]
        new_tent = [row[3] for row in tent]

        for i in range(len(nbhd_tent)):

            # Consider tentative DROP moves to add to final list
            sol_cand = _sol_move(drop=nbhd_tent[i])

            # Process new solutions
            if new_tent[i] == True:
                # Calculate constraints for new solutions
                cand_feas = _con_update(sol_cand, obj_tent[i])
                nw += 1
            else:
                # Otherwise the constraint is definitely feasible
                cand_feas = 1
                lu += 1

            if cand_feas > 0:
                # Add feasible solutions to neighborhood
                nbhd_drop.append(nbhd_tent[i])
                obj_drop.append(obj_tent[i])
            else:
                # Skip infeasible solutions
                continue

            if len(nbhd_drop) >= nbhd_drop_lim2:
                # Break if we've found enough DROP neighbors
                break

        Log.nbhd_list("DROP", nbhd_drop)

        #----------------------------------------------------------------------
        # Handle the event of an unsuccessful search

        if (len(nbhd_add) == 0) and (len(nbhd_drop) == 0):
            # If no moves are found, search for more
            repeat = True
            Log.write("No neighbors found. Search repeating.")
            if (len(cand_add) == 0) and (len(cand_drop) == 0):
                # If no candidates remain, start removing tabu rules
                Log.write("Dead end. Removing tabu rules.")
                cand_add = [i for i in range(len(sol))]
                cand_drop = [i for i in range(len(sol))]
                nonimp_out += 1
                mina = np.inf
                mind = np.inf
                if tabu_add.count(0) < len(tabu_add):
                    # If there exists a nonzero ADD tenure, find the smallest
                    ta = np.array(tabu_add)
                    mina = tabu_add.index(np.min(ta[ta > 0])) # min !=0 tenure
                if tabu_drop.count(0) < len(tabu_drop):
                    # If there exists a nonzero DROP tenure, find the smallest
                    td = np.array(tabu_drop)
                    mind = tabu_drop.index(np.min(td[td > 0])) # min !=0 tenure
                if mina <= mind:
                    tabu_add[mina] = 0
                else:
                    tabu_drop[mind] = 0

    #--------------------------------------------------------------------------
    # SWAP move selection (only after a successful ADD/DROP search)

    if (len(nbhd_add) > 0) and (len(nbhd_drop) > 0):
        # SWAP moves are only defined if we have ADD and DROP moves to combine
        limit = min(len(nbhd_add), len(nbhd_drop))
        i = 0 # current ADD index
        while (len(nbhd_swap) < nbhd_swap_lim) and (i < limit):
            # Iterate through all ADD moves up to the limit
            for j in range(i+1):
                # Iterate through all DROP moves up to the current ADD move
                if vtype[nbhd_add[i]] != vtype[nbhd_drop[j]]:
                    # Skip SWAPs with different vehicle types
                    continue
                if nbhd_add[i] == nbhd_drop[j]:
                    # Skip SWAPs with the same ADD and DROP values
                    continue
                sol_cand = _sol_move(add=nbhd_add[i], drop=nbhd_drop[j])
                cand_feas, _, cand_obj, _ = _obj_lookup(sol_cand) # look up sol
                if cand_feas >= 0:
                    # Successful lookup
                    lu += 1
                if cand_feas < 0:
                    # Calculate constraints for new solutions
                    cand_feas = _con_update(sol_cand, cand_obj)
                    nw += 1
                if cand_feas > 0:
                    # Add feasible solutions
                    nbhd_swap.append([nbhd_add[i], nbhd_drop[j]])
                    obj_swap.append(cand_obj)
                if len(nbhd_swap) >= nbhd_swap_lim:
                    # Break early if we've already reached the limit this loop
                    break
            i += 1

    Log.nbhd_list("SWAP", nbhd_swap)

    #--------------------------------------------------------------------------
    # Find and return the two best moves

    obj_best1 = np.inf
    obj_best2 = np.inf
    move_best1 = [-1, -1]
    move_best2 = [-1, -1]

    # ADD moves
    for i in range(len(nbhd_add)):
        if obj_add[i] < obj_best1:
            obj_best2 = obj_best1
            obj_best1 = obj_add[i]
            move_best2 = move_best1[:]
            move_best1 = [nbhd_add[i], -1]
        elif obj_add[i] < obj_best2:
            obj_best2 = obj_add[i]
            move_best2 = [nbhd_add[i], -1]

    # DROP moves
    for i in range(len(nbhd_drop)):
        if obj_drop[i] < obj_best1:
            obj_best2 = obj_best1
            obj_best1 = obj_drop[i]
            move_best2 = move_best1[:]
            move_best1 = [-1, nbhd_drop[i]]
        elif obj_drop[i] < obj_best2:
            obj_best2 = obj_drop[i]
            move_best2 = [-1, nbhd_drop[i]]

    # SWAP moves
    for i in range(len(nbhd_swap)):
        if obj_swap[i] < obj_best1:
            obj_best2 = obj_best1
            obj_best1 = obj_swap[i]
            move_best2 = move_best1[:]
            move_best1 = nbhd_swap[i][:]
        elif obj_swap[i] < obj_best2:
            obj_best2 = obj_swap[i]
            move_best2 = nbhd_swap[i][:]

    Log.nbhd_stats(lu, nw, time()-start)

    # Return the two best solutions
    return ([move_best1[0], move_best1[1], obj_best1, vtype[move_best1[0]]],
            [move_best2[0], move_best2[1], obj_best2, vtype[move_best2[0]]])

#------------------------------------------------------------------------------
def _sol_move(add=-1, drop=-1):
    """Returns the solution vector resulting from a specified ADD and/or DROP.

    The keyword arguments 'add' and 'drop' indicate the indices of the altered
    variables, defaulting to -1 to indicate no move.
    """

    sol_move = sol[:]
    if add >= 0:
        sol_move[add] += step
    if drop >= 0:
        sol_move[drop] -= step
    return sol_move

#------------------------------------------------------------------------------
def _tenure_increment():
    """Updates the tabu tenures."""

    #################################################################### The paper uses random increments on [0.01, 0.09].

    global tenure, tenure_factor
    tenure *= tenure_factor

#------------------------------------------------------------------------------

def _cooling(iteration):
    """Updates the SA temperature."""

    ##################################################################### Consider other cooling schedules.

    global tmp, tmp_factor
    tmp *= tmp_factor

#------------------------------------------------------------------------------
def _exhaustive_search(sol_start, obj_start, swaps=True):
    """Conducts an exhaustive search from a given starting solution.

    Beginning from a given starting solution, iteratively moves to the best
    feasible neighbor until at a local optimum. Tabu rules are ignored but
    constraints are enforced.

    This is an expensive process since every possible neighbor must be
    considered. As such, it is meant to be used only at the end of the main
    search process to at least guarantee local optimality.

    Accepts the following optional keyword arguments:
        swaps -- Whether or not to consider SWAP moves. Defaults to True. Set
            to False to limit the size of the neighborhood in case the search
            is taking too long.
    """

    global sol, obj, sol_best, obj_best, vcurrent, Log

    Log.exhaustive_header()

    sol = sol_start[:]
    obj = obj_start

    while True:
        # Continue iterative process until local optimality causes a break

        best_move = _best_neighbor(sol, obj, swaps=swaps) # find best neighbor

        if (best_move[0] < 0) and (best_move[1] < 0):
            # Break if the best move is [-1,-1], indicating no move
            Log.write("Reached local optimality!")
            break

        # Otherwise, update the current solution
        sol = _sol_move(add=best_move[0], drop=best_move[1])
        obj = best_move[2]
        Log.local_move(best_move[0:2], best_move[2])
        Log.obj_append("end", obj, obj)

        # Update current vehicles in use
        if best_move[0] >= 0:
            vcurrent[vtype[best_move[0]]] += 1
        if best_move[1] >= 0:
            vcurrent[vtype[best_move[1]]] -= 1

        vcurrent = [0 for i in range(len(vbound))]
        for i in range(len(sol)):
            vcurrent[vtype[i]] += sol[i]

        if obj < obj_best:
            # Update best known solution
            sol_best = sol[:]
            obj_best = obj

#------------------------------------------------------------------------------
def _best_neighbor(sol, obj, swaps=True):
    """Returns the best feasible neighbor of a given solution.

    Accepts a solution and objective value and returns the neighboring move
    with the lowest objective value that is lower than the current objective.
    Tabu rules are ignored but constraints are enforced.

    Returns a list of the form [add, drop, obj], where the first two elements
    indicate the indices of the ADD and DROP element and the last indicates
    the objective of the indicated move. An index of -1 indicates no change,
    and so a result with add=-1 and drop=-1 indicates local optimality.

    Accepts the following optional keyword arguments:
        swaps -- Whether or not to consider SWAP moves. Defaults to True. Set
            to False to limit the size of the neighborhood in case the search
            is taking too long.
    """

    # Best neighbor objective
    obj_best1 = np.inf
    sol_best1 = [-1, -1]

    # Consider every possible ADD
    for choice in range(len(sol)):
        if (sol[choice] + step) > ub[choice]:
            # Don't attempt an ADD that would exceed an upper bound
            continue
        if vcurrent[vtype[choice]] + step > vcapacity[vtype[choice]]:
            # Don't attempt an ADD that would exceed the vehicle limit
            continue
        sol_cand = _sol_move(add=choice)
        feas_cand, _, obj_cand, _ = _obj_lookup(sol_cand) # look up solution
        if feas_cand != 0:
            # Rule out moves known to be infeasible
            if obj_cand < min(obj, obj_best1):
                # Rule out moves that do not improve the objective
                if feas_cand > 0:
                    # Immediately keep a known feasible candidate
                    sol_best1 = [choice, -1]
                    obj_best1 = obj_cand
                else:
                    # Test feasibility of an unknown candidate
                    feas_cand = _con_update(sol_cand, obj_cand)
                    if feas_cand > 0:
                        sol_best1 = [choice, -1]
                        obj_best1 = obj_cand

    # Consider every possible DROP
    for choice in range(len(sol)):
        if (sol[choice] - step) < lb[choice]:
            # Don't attempt a DROP that would fall below a lower bound
            continue
        sol_cand = _sol_move(drop=choice)
        feas_cand, _, obj_cand, _ = _obj_lookup(sol_cand) # look up solution
        if feas_cand != 0:
            # Rule out moves known to be infeasible
            if obj_cand < min(obj, obj_best1):
                # Rule out moves that do not improve the objective
                if feas_cand > 0:
                    # Immediately keep a known feasible candidate
                    sol_best1 = [-1, choice]
                    obj_best1 = obj_cand
                else:
                    # Test feasibility of an unknown candidate
                    feas_cand = _con_update(sol_cand, obj_cand)
                    if feas_cand > 0:
                        sol_best1 = [-1, choice]
                        obj_best1 = obj_cand

    # Consider every possible SWAP
    if swaps == True:
        for choice_add in range(len(sol)):
            for choice_drop in range(len(sol)):
                if vtype[choice_add] != vtype[choice_drop]:
                    # Skip SWAPs with different vehicle types
                    continue
                if choice_add == choice_drop:
                    # Skip cases with the same ADD and DROP indices
                    continue
                if (sol[choice_add] + step > ub[choice_add]):
                    # Skip ADDs that exceed an upper bound
                    continue
                if (sol[choice_drop] - step < lb[choice_drop]):
                    # Skip DROPs that fall below a lower bound
                    continue
                sol_cand = _sol_move(add=choice_add, drop=choice_drop)
                feas_cand, _, obj_cand, _ = _obj_lookup(sol_cand) # look up sol
                if feas_cand != 0:
                    # Rule out moves known to be infeasible
                    if obj_cand < min(obj, obj_best1):
                        # Rule out moves that do not improve the objective
                        if feas_cand > 0:
                            # Immediately keep a known feasible candidate
                            sol_best1 = [choice_add, choice_drop]
                            obj_best1 = obj_cand
                        else:
                            # Test feasibility of an unknown candidate
                            feas_cand = _con_update(sol_cand, obj_cand)
                            if feas_cand > 0:
                                sol_best1 = [choice_add, choice_drop]
                                obj_best1 = obj_cand

    return [sol_best1[0], sol_best1[1], obj_best1]

#------------------------------------------------------------------------------
def _obj_lookup(sol):
    """Retrieves or generates the objective value of a given solution.

    Uses the solution log's methods to look up a given solution. If the
    solution is found in the log, then we retrieve its objective. If not,
    then we generate its objective and add a placeholder to the log.

    In either case we return the feasibility status, the constraints (or a
    placeholder if unknown), and the objective of the solution.
    """

    global SolLog

    # Find out whether the solution has been logged
    logged = SolLog.search(sol)

    if logged == True:
        # Logged and ready for lookup
        cand_feas, cand_con, cand_obj, cand_obj2 = SolLog.lookup(sol)
    else:
        # New, so we create a placeholder
        cand_con = [0 for i in range(con_length)] # placeholder constraints
        cand_feas = -1 # unknown feasibility
        start = time()
        cand_obj, cand_obj2 = Obj.calculate(sol) # calculate objective
        cand_obj_time = time() - start # total time to calculate objective
        cand_con_time = 0 # unknown time to calculate constraints
        # Create new row with tentative information
        SolLog.create(sol, cand_feas, cand_con, cand_con_time, cand_obj,
                      cand_obj2, cand_obj_time)

    return cand_feas, cand_con, cand_obj, cand_obj2

#------------------------------------------------------------------------------
def _con_update(sol, obj):
    """Generates (or updates) the constraints for a given solution.

    Accepts a solution vector and objective value. We calculate the constraint
    functions for the given solution and then look it up in the solution log.

    This method is only called for solutions that have already been logged (or
    that at least have a placeholder row), and so we use the results to update
    the solution's row.

    After completing this update, the value of the feasibility status is
    returned (1 for feasible, 0 for infeasible).
    """

    global SolLog

    # Calculate the constraint functions
    start = time()
    cand_feas, cand_con = Con.calculate(sol)
    cand_con_time = time() - start # total time to evaluate constraint function

    # Already logged, so update its information
    SolLog.update(sol, feas=cand_feas, con=cand_con, con_time=cand_con_time)

    return cand_feas

#==============================================================================
# Execution
#==============================================================================

driver(display=True, max_iter=10, exhaustive=False, swaps=False, percent=0.01,
       pickup=False)
