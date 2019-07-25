#include <deque>
#include <vector>

#include "bfcheck.hpp"

using std::deque;
using std::vector;

  BF_Checker::BF_Checker(vector< vector < int > > graph, 
	     vector< vector< double > > constraints, 
	     int k, double gamma): _graph(graph), 
	     _constraints(constraints), _gamma(gamma) {
	       
	       this->_feasible = false;
	       this->_first = true;
	       int n = graph.size();
	       this->bf(n, k);
  }
  
  bool BF_Checker::is_feasible(void) {
    return this->_feasible;
  }
  
  double BF_Checker::get_best(void) {
    return this->_best;
  }

  void BF_Checker::bf(int n, int k) {
    deque<int> s;
    vector< vector< bool > > clusters(k, vector< bool >(n));
    vector< int > sizes(k, 0);

    do {

      while(s.size() < n)
	s.push_back(0);
      
      // s is the current solution
      sizes.assign(k, 0);
      for(int i=0; i<k; i++)
	for (int j=0; j<n; j++)
	  if(s[j] == i) {
	    clusters[i][j] = true;
	    sizes[i]++;
	  }
	  else 
	    clusters[i][j] = false;
	  
      int min_size = sizes[0];
      for (int i=1; i<k; i++)
	if (sizes[i] < min_size)
	  min_size = sizes[i];
	
      bool all_connected = true;
      for (int i=0; all_connected && i<k; i++)
	all_connected &= is_connected(clusters[i]);
      
      if(all_connected) {
	double value = min_size - _gamma * get_penalty(s);
	if (_first) {
	  _best = value;
	  _feasible = true;
	  _first = false;
	}
	else if (value > _best)
	  _best = value;
      }
      
      int t = s.back();
      s.pop_back();
      while (!s.empty() && t == k-1) {
	t = s.back();
	s.pop_back();
      }
      
      if (t<k-1)
	s.push_back(t + 1);
    } while (!s.empty());
  }

  bool BF_Checker::is_connected(const vector< bool >& cluster) {
    int n = cluster.size();
    _visited.assign(n, false);
    
    deque<int> s;
    for(int i=0; s.empty() && i<n; i++)
      if (cluster[i]) {
	s.push_back(i);
	_visited[i] = true;
       }
    
    while(!s.empty()){
      int u = s.back();
      s.pop_back();
      for (int i=0, sz=_graph[u].size(); i<sz; i++) {
        int v  = _graph[u][i];
	if (cluster[v] and !_visited[v]) {
	  s.push_back(v);
	  _visited[v] = true;
	}
      }
    }
    
    bool connected = true;
    for(int i=0; connected && i<n; i++)
      if(cluster[i] && !_visited[i])
	connected = false;
      
    return connected;
  }
  
  double BF_Checker::get_penalty(const deque<int>& clusters) {
    double penalty = 0;
    for (int i=0, sz=_constraints.size(); i<sz; i++) {
      int first = _constraints[i][0];
      int second = _constraints[i][1];
      double weight = _constraints[i][2];
      if (clusters[first] == clusters[second])
	penalty += weight;
    }
    return penalty;
  }
  
  