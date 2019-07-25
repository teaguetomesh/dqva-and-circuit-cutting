Important Notice
----------------
Before importing the `comparison_utils` module, you have to build the code in the `../brute-force` directory. The necessary instructions are included there. 


Description
-----------
The notebook `compare.ipynb` shows how to compare the results of the MIP solver and the brute-force algorithm on an instance. You just need to call the `compare` function (imported from `comparsion_utils`). It will make the necessary format conversions, calls the two algorithms and compares the results. 

These are the arguments of the function `compare(graph, constraints, k, gamma)`:
- `graph`: a list of tuples. Each tuple is an edge. The vertices can be string, integer, or any other object that supports equality test. 
- `constraints`: a list of lists. Each inner list has three elements: the first node, the second node, and the weight of the corresponding can-not-link constraint. 
- `k`: number of clusters
- `gamma`: the parameter for balancing the two components of the objective function

