# social-transit-solver-python

A deprecated Python implementation of the main tabu search/simulated annealing hybrid solution algorithm for use in a research project of mine dealing with a public transit design model with social access objectives.

This was an early version of the main solver eventually implemented in C++, located [here](https://github.com/adam-rumpf/social-transit-solver). It was not directly used for any results in the paper but it did prove useful in testing certain aspects of the solution algorithm. Note also that it represents a slightly different model from the final version in the paper, and as such it includes a different set of input files from the main solver. Also note that this module requires the use of the [CPLEX](https://www.ibm.com/analytics/cplex-optimizer) Python API for solving the linear programs involved in some of the submodules.

I would not expect this program to be of use to anyone outside of my research group, but it is being made available if anyone is interested.

## Output Logs

This program writes outputs to a local `log/` folder. The following files are produced:

* `event.txt`: A log explaining what occurred during each iteration of the solution process, including the contents of the search neighborhood, the time spent searching, the selected move, the current objective value, and other events.
* `final.txt`: Includes the best known solution vector along with its objective value.
* `memory.txt`: The memory structures associated with the tabu search/simulated annealing hybrid search process. Used to continue a halted search process.
* `metrics.txt`: Accessibility metrics of each population center for the best known solution.
* `objective.txt`: Log of the current and best objective values in each iteration of the search process.
* `solution.txt`: Log of all previously-searched solutions along with their feasibility status, constraint function elements, objective values, and evaluation times. Used to maintain a solution dictionary in order to avoid having to process searched solutions a second time.

## Data Folder

This program reads input files from a local `data/` folder. The following data files should be included in this folder:

* [Arcdata_Con.txt](#arcdata_contxt)
* [Arcdata_Obj.txt](#arcdata_objtxt)
* [Names_Con.txt](#names_contxt)
* [Names_Line.txt](#names_linetxt)
* [Names_Obj.txt](#names_objtxt)
* [Nodedata_Con.txt](#nodedata_contxt)
* [Nodedata_Obj.txt](#nodedata_objtxt)
* [ODdata.txt](#oddatatxt)
* [Problem_Data.txt](#problem_datatxt)
* [Search_Parameters.txt](#search_parameterstxt)
* [Transitdata.txt](#transitdatatxt)

The contents of these files will be explained below. Most include IDs for each of their elements. For the purposes of our solution algorithm these are assumed to consecutive integers beginning at `0`, and this is how they will be treated for the purposes of array placement.

This version of the solution algorithm included two completely separate network files: one for use in objective evaluation (which includes population centers and facilities), and one for use in constraint evaluation (which in general is a subset of the objective network). This was redone in the final version by simply using arc type IDs to specify which arcs should be involved in which calculations.

Unless otherwise specified, the following units are used:

* cost = dollars
* distance = miles
* time = minutes

### `Arcdata_Con.txt`

Defines the arcs for the Spiess version of the transit network, which includes day-to-day traffic origins and destinations. For use in the constraint module.

IMPORTANT: The constant-cost [Spiess and Florian model](https://www.researchgate.net/publication/222385476_Optimal_Strategies_A_New_Assignment_Model_for_Transit_Networks) involves a modified minimum cost flow LP that will not work unless there exists a path from every origin to every destination.

The columns are arranged as follows:
* `ArcID`: Index of arc. Should be consecutive integers starting at `0`.
* `ArcType`: `0` for line arcs, `1` for boarding arcs, and `-1` otherwise. This is to highlight the arcs whose travel time needs to be updated with the solution.
* `Line`: Index of line associated with the arc (`-1` if N/A).
* `Out`: Index of arc tail (node that the arc leaves).
* `In`: Index of arc head (node that the arc enters).
* `TrTime`: Constant time to traverse arc. Mostly used for line arcs to represent the time required to drive between stops.

### `Arcdata_Obj.txt`

Defines the arcs for a simpler version of the transit network for calculating pairwise travel times. Includes the origins and destinations of interest for the objective function. The only auxiliary nodes and arcs at a stop are included to separate boarding time from travel time.

IMPORTANT: The shortest path LP defined by this network will not work unless there exists a path from every origin to every destination and vice-versa. For this reason, this network may involve many more walking arcs than the Spiess version of the network. Make sure to also make every walking arc to and from an origin/destination node reversible.

The columns are arranged as follows:
* `ArcID`: Index of arc. Should be consecutive integers starting at `0`.
* `ArcType`: `0` for line arcs, `1` for boarding arcs, and `-1` otherwise. This is to highlight the arcs whose travel time needs to be updated with the solution.
* `Line`: Index of line associated with the arc (`-1` for walking arcs).
* `Out`: Index of arc tail (node that the arc leaves).
* `In`: Index of arc head (node that the arc enters).
* `TrTime`: Constant time to traverse arc. Mostly used for line arcs to represent the time required to drive between stops, or walking times for walking arcs.

### `Names_Con.txt`

Lists the name associated with each node of the network. This serves as a guide to which node IDs correspond to which real-world locations, since the node IDs may be changed during the network building process. It is not used by the main solution algorithm but may prove useful in the results analysis.

The artificial boarding nodes are named by appending their line's name onto their stop's name. Origin/destination nodes for stops are name by appending the word "origin" or "destination" onto the stop's name.

### `Names_Line.txt`

Lists the name associated with each transit line. This serves as a guide to which line IDs correspond to which real-world lines, since the line IDs may be changed during the network building process. It is not used by the main solution algorithm but may prove useful in the results analysis.

### `Names_Obj.txt`

Lists the name associated with each node of the network. This serves as a guide to which node IDs correspond to which real-world locations, since the node IDs may be changed during the network building process. It is not used by the main solution algorithm but may prove useful in the results analysis.

The artificial boarding nodes are named by appending their line's name onto their stop's name.

### `Nodedata_Con.txt`

Defines the nodes for the Spiess version of the transit network, which includes day-to-day traffic origins and destinations.

The columns are arranged as follows:
* `NodeID`: Index of node. Should be consecutive integers starting at `0`.
* `NodeType`: `0` for origin, `1` for destination, `2` for stop, `3` for line.
* `Line`: Index of line associated with the node (`-1` if N/A).
* `Platform`: Index of stop associated with the node (`-1` if N/A).

### `Nodedata_Obj.txt`

Defines the nodes for a simpler version of the transit network for calculating pairwise travel times. Includes the origins and destinations of interest for the objective function. The only auxiliary nodes and arcs at a stop are included to separate boarding time from travel time.

The columns are arranged as follows:
* `NodeID`: Index of node. Should be consecutive integers starting at `0`.
* `NodeType`: `0` for origin (population center), `1` for destination (facility), `2` for stop, `3` for line
* `Line`: Index of line associated with the node (`-1` if N/A).
* `Val`: Value associated with an origin or destination (`-1` if N/A). For a population center this is the population, while for a facility this is the capacity.

### `ODdata.txt`

Defines the OD pairs for the Spiess version of the network.

The columns are arranged as follows:
* `ID`: Index of OD pair. Should be consecutive integer starting at `0`.
* `Origin`: Index of origin node.
* `Dest`: Index of destination node.
* `ODVol`: Travel demand for the given OD pair.

### `Problem_Data.txt`

Input data related to the problem definition, including the initial solution, variable bounds, line data, constraint parameters, etc. Each parameter is written on a separate row, with vectors being written as a single tab-separated row.

Several vectors correspond to the set of lines, while others correspond to the set of vehicle types. In both cases we always maintain a consistent order, so that the elements of multiple vectors with the same index correspond to each other.

The rows are written in the following order:
* `vbound`: Vector of bounds for each vehicle type (i.e. limit to how many of each vehicle type is available).
* `vcapacity`: Vector of capacities for each vehicle type (number of passengers).
* `vcost`: Vector of operating costs for each vehicle type (dollars to operate for the entirety of the day-to-day time horizon).
* `vfare`: Vector of fares to board each vehicle type.
* `operator_init`: Initial operator cost, for use in the `OperatorCost()` bound.
* `user_init`: Initial user cost, for use in the `UserCost()` bound.
* `con_length`: Number of constraint function elements. This should be a constant `5` for the purposes of our study.
* `riding_weight`: Weight of total in-vehicle travel time term in `UserCost()` function. Note that this includes only the actual time, not the modified time from the nonlinear model.
* `walking_weight`: Weight of total walking travel time term in `UserCost()` function. Note that this includes all arcs with index `-1`, which is everything except for boarding and line arcs.
* `waiting_weight`: Weight of total waiting time term in `UserCost()` function.
* `conical_alpha`: Parameter for conical volume delay function.
* `time_horizon`: Time horizon (minutes) for frequency-based day-to-day traffic.
* `spiess_epsilon`: Threshold for early termination of nonlinear Spiess model, as an absolute optimality gap threshold. Note that this value should be chosen based on what seems reasonable given the expected magnitude of the nonlinear model objective, which may in general be enormous.
* `spiess_max`: Maximum number of iterations of the nonlinear Spiess model.
* `cplex_epsilon`: Variable threshold for the CPLEX cleanup method. Values calculated by CPLEX during its solves that fall below this threshold are deleted between solves, which can help to clean up error buildup and speed things up.
* `fca_cutoff`: Time cutoff (minutes) to use in calculating 2SFCA catchment areas.

### `Search_Parameters.txt`

Contains parameters for the TS/SA hybrid algorithm. Each parameter is written on a separate row, arranged in the following order:
* Comment row.
* `tmp_init`: Initial SA temperature.
* `tmp_factor`: Factor by which to multiply the temperature when cooling. Should be between `0` and `1`. May eventually replace with different parameters for use in a different type of cooling schedule.
* `attractive_max`: Maximum size of attractive solution set.
* `nbhd_add_lim`: Number of ADD moves to collect during first pass.
* `nbhd_add_lim2`: Actual number of desired ADD moves after second pass (should be less than the previous limit).
* `nbhd_drop_lim`: Number of DROP moves to collect during first pass.
* `nbhd_drop_lim2`: Actual number of desired DROP moves after second pass (should be less than the previous limit).
* `nbhd_swap_lim`: Actual number of desired SWAP moves.
* `tenure_init`: Initial tabu tenure.
* `tenure_factor`: Factor by which to multiply the tabu tenures when increasing. May eventually implement a randomized system.
* `nonimp_in_max`: Maximum value of inner nonimprovement counter.
* `nonimp_out_max`: Maximum value of outer nonimprovement counter.
* `step`: Increment of ADD/DROP moves.

### `Transitdata.txt`

Defines the transit lines of the network.

The columns are arranged as follows:
* `LineID`: Index of line.
* `Freq`: Initial line frequency. Note that this will be recalculated many times during the solution algorithm, but the updated frequencies will be maintained in memory.
* `Capacity`: Initial line capacity. As with `Freq`, this will also be recalculated during the solution algorithm.
* `lb`: Vector of variable lower bounds (same order as solution vector).
* `ub`: Vector of variable upper bounds (same order as solution vector).
* `turnaround`: Vector of turnaround time (in minutes) for each route (same order as solution vector). This is the amount of time that passes for a vehicle between consecutive completions of a circuit, and includes both layover time and time required to drive back to the route start.
* `sol_init`: Initial solution vector (same order as solution vector).
* `vtype`: Vector of vehicle types for each line (same order as solution vector).
