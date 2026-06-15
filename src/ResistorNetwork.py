import numpy as np
from scipy.optimize import root, least_squares
from inspect import isfunction
import random
import itertools
import networkx as nx  
from networkx.algorithms.bridges import bridges


## use k=1000.0 -> mS to boost current signal!
minimum_cond = 1e-5

# Conductance function for MOSFET-like element
def G_MOS(G, Vsrc, Vdrn, VT=0.7, k=1000.0):
    Vbar = 0.5 * (Vsrc + Vdrn)
    g = G - VT - Vbar
    return k*np.clip(g, minimum_cond,None)

def G_MOS_grad(G, Vsrc, Vdrn, VT=0.7, k=1000.0):
    """
    Returns (dg/dVsrc, dg/dVdrn) for the MOS conductance G_MOS.
    """
    # compute the raw argument before clipping
    Vbar      = 0.5 * (Vsrc + Vdrn)
    g_raw     = G - VT - Vbar
    # mask of where we're in the linear (unclipped) region
    lin_mask  = (g_raw > minimum_cond)

    # vectorize. when lin_mask=True, derivative = -k/2; else 0.
    dg_dVsrc = (-0.5 * k) * lin_mask
    dg_dVdrn = (-0.5 * k) * lin_mask
    return dg_dVsrc, dg_dVdrn


# Conductance function for MOSFET-like element
def G_MOS_NORM(G, Vsrc, Vdrn, VT=0.7, k=1000.0):
    Vbar = 0.5 * (Vsrc + Vdrn)
    g = np.exp(G) - VT - Vbar
    return k*np.clip(g, minimum_cond,None)

# Conductance function for smoothly variable resistor and ideal diode (not possible)
def G_DIO(G, Vp, Vm, k=1000.0):
    g = G * (Vp>Vm)
    return k*np.clip(g,minimum_cond,None)

# Conductance function for X9C303S8I digital potentiometer (shorturl.at/mEbBy)
def G_DIG(G, Vp, Vm, k=1000.0):
    r_min, r_max, steps = 40.0, 32000.0, 100 #min/max resistance, number of states
    return (k/ r_min) * (r_min / r_max) ** (np.round(G) / (steps - 1)) # conductance (not resistance)


class ResistorNetwork:
    def __init__(self, node_from, node_to, conductance_funcs, params=None):
        """
        node_from, node_to: lists of length E
        conductance_funcs: list of callables, each func(param, Vsrc, Vdrn)->conductance
        params: array of length E, each element's tunable parameter
        num_nodes: total number of nodes N
        """
        self.node_from = np.array(node_from, dtype=int)
        self.node_to   = np.array(node_to,   dtype=int)
        self.funcs     = conductance_funcs
        self.funcs_grad = None

        if params is None:
            self.params = np.array([1.0]*np.size(self.node_from))
        elif len(params)==2 and len(node_from) != 2:
            # this is for giving mean / std for random param initialization
            self.params= params[0] + np.random.randn(np.size(self.node_from))*params[1]
        else:
            self.params = np.array(params, dtype=float)
  
        # Pre-assemble directed edge pairs
        self.edges = np.vstack((self.node_from, self.node_to)).T
        self.num_nodes = np.max(self.edges)+1

        # if just one function or label given, repeat for every edge
        if not isinstance(self.funcs, list):
            self.funcs = [self.funcs]*np.size(self.node_from)
            self.funcs_grad = [None]*np.size(self.node_from)

        for idx,i in enumerate(self.funcs):
            if not isfunction(i):
                if isinstance(i,str):
                    self.funcs_grad[idx] = None

                    if i.lower() == 'g_mos': #MOSFET
                        self.funcs[idx] = G_MOS
                        self.funcs_grad[idx] = G_MOS_grad
                    elif i.lower() == 'g_dio': #resistor + diode
                        self.funcs[idx] = G_DIO
                    elif i.lower() == 'g_dig': #potentiometer
                        self.funcs[idx] = G_DIG
                    elif i.lower() == 'g_mos_norm': #potentiometer
                        self.funcs[idx] = G_MOS_NORM
                    else:
                        raise ValueError("String does not match stored edge function")
           
                else:
                    raise ValueError("Conductance_funcs must be fn or string!")                
    
    def solve_equilibrium(
        self,
        fixed_nodes,
        fixed_values,
        initial_guess=None,
        usebounds=False,         # new flag to switch bounds on/off
        max_nfev=1000,
        tol=1e-8,
        use_analytic_jac=True,  # keep analytic-Jac option
    ):
        """
        Solve for node voltages with optional box bounds taken from fixed_values.
        - fixed_nodes: indices of nodes held at fixed_values
        - fixed_values: array of voltages for fixed_nodes
        - usebounds: if [min,max], use them!
        """
        # 1) mark fixed vs free
        fixed = np.zeros(self.num_nodes, dtype=bool)
        fixed[fixed_nodes] = True
    
        # 2) build base voltage array
        V_fixed = np.zeros(self.num_nodes)
        V_fixed[fixed_nodes] = fixed_values
    
        # 3) initial guess
        if initial_guess is None:
            V0 = np.full(self.num_nodes, np.mean(fixed_values))
        else:
            V0 = initial_guess.astype(float).copy()
        V0_free = V0[~fixed]
    
        # 4) bounds logic
        if np.size(usebounds)==2:
            vmin, vmax = usebounds
            lower_bounds = vmin * np.ones(self.num_nodes)
            upper_bounds = vmax * np.ones(self.num_nodes)
        else:
            lower_bounds = -np.inf * np.ones(self.num_nodes)
            upper_bounds =  np.inf * np.ones(self.num_nodes)
    
        lb_free = lower_bounds[~fixed]
        ub_free = upper_bounds[~fixed]

        # 4b) build a global→free index map once
        free_idx = np.where(~fixed)[0]                   # array of global node indices
        global2free = {node: pos for pos, node in enumerate(free_idx)}

    
        # 5) residual function
        def mismatch(V_var):
            V = V_fixed.copy()
            V[~fixed] = V_var
            I = np.zeros_like(V)
            for idx, (i, j) in enumerate(self.edges):
                g = self.funcs[idx](self.params[idx], V[i], V[j])
                I[i] += g * (V[j] - V[i])
                I[j] += g * (V[i] - V[j])
            return I[~fixed]
    
       
        # 6) analytic Jacobian (optional)
        def mismatch_jac(V_var):
            # reconstruct full voltages
            V = V_fixed.copy()
            V[~fixed] = V_var
        
            n_free = free_idx.size
            J = np.zeros((n_free, n_free))
        
            for idx_edge, (i, j) in enumerate(self.edges):
                g = self.funcs[idx_edge](self.params[idx_edge], V[i], V[j])
                dg_dVi, dg_dVj = self.funcs_grad[idx_edge](self.params[idx_edge], V[i], V[j])
        
                ii = global2free.get(i)    # None if i is fixed
                jj = global2free.get(j)    # None if j is fixed
        
                # contributions to ∂I_i/∂V
                if ii is not None:
                    J[ii, ii] += -g + dg_dVi * (V[j] - V[i])
                    if jj is not None:
                        J[ii, jj] +=  g + dg_dVj * (V[j] - V[i])
        
                # contributions to ∂I_j/∂V
                if jj is not None:
                    J[jj, jj] += -g + dg_dVj * (V[i] - V[j])
                    if ii is not None:
                        J[jj, ii] +=  g + dg_dVi * (V[i] - V[j])
        
            return J
    
        # 7) solve with bounds
        sol = least_squares(
            mismatch,
            V0_free,
            jac=mismatch_jac if use_analytic_jac else '2-point',
            bounds=(lb_free, ub_free),
            xtol=tol,
            ftol=tol,
            max_nfev=max_nfev,
            method='trf',
            verbose=0
        )
    
        # 8) assemble full solution
        V_sol = V_fixed.copy()
        V_sol[~fixed] = sol.x
        return V_sol
    

    def display_graph(self, directed: bool = False,nodegroups=[],figsize=(6,6),ax=None):
        """
        Display the network:
          - Nodes are numbered circles.
          - If directed=True, edges are arrows.
          - Reverse‐direction arrows between the same pair are offset.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        colors = ['darkred','darkblue','darkgreen','darkyellow']
    
        # 1) Build the graph
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
    
        # Add every node found in node_from/node_to
        all_nodes = set(self.node_from) | set(self.node_to)
        G.add_nodes_from(all_nodes)
    
        # Add edges
        edges = list(zip(self.node_from, self.node_to))
        G.add_edges_from(edges)
    
        # 2) Choose a layout
        pos = nx.spring_layout(G,iterations=1000)

        # 3) Draw nodes
        if ax is None:
            plt.figure(figsize=figsize);
        else:
            plt.sca(ax);

        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=600,
            node_color="lightblue",
            linewidths=1,
            edgecolors="black"
        )
        # 3.5) highlight any node groups
        for idx,subset in enumerate(nodegroups):
             nx.draw_networkx_nodes(
            G, pos,
            node_size=600,
            node_color=colors[idx],
            linewidths=1,
            edgecolors="black",
            nodelist=subset
        )

        # Label each node with its number
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_color="black", font_size=12)
    
        # 4) Draw edges
        if not directed:
            # simple undirected edges
            nx.draw_networkx_edges(G, pos, width=2)
        else:
            # for directed, draw each arrow; offset if there is an edge back the other way
            edge_set = set(edges)
            drawn = set()
            for u, v in edges:
                if (u, v) in drawn:
                    continue
                # if there's also a v->u edge, offset one arc up and one down
                if (v, u) in edge_set:
                    # choose consistently which way to bend
                    rad = 0.1 if u < v else -0.1
                else:
                    rad = 0.0
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    arrowstyle='-|>',
                    arrowsize=16,
                    width=2,
                    connectionstyle=f'arc3,rad={rad}'
                )
                drawn.add((u, v))
    
        plt.axis('off');
        plt.tight_layout();
       # plt.show();



def makeNetwork(network_type, spec_vec,conduct_func, directed = False,params=None):
    """ Single function to create networks of varying topologies and sizes"""

    typelist = ['PLATTICE','LATTICE','DENSE','RANDOM','PDENSE']
    typefns = [periodic_lattice,nonperiodic_lattice, dense_network, random_network,dense_network_periodic]

    # generate desired network type and structure
    node_from, node_to = typefns[typelist.index(network_type)](spec_vec)

    if directed: # duplicate all edges but backwards
        node_from, node_to  = node_from + node_to, node_to + node_from

    return ResistorNetwork(node_from, node_to, conduct_func, params=params)

def periodic_lattice(dimensionvec):
    return lattice_network(dimensionvec, periodic=True)

def nonperiodic_lattice(dimensionvec):
    return lattice_network(dimensionvec, periodic=False)
    
def lattice_network(dimensionvec, periodic=False):
    """Generates a (optionally periodic) square lattice network."""
    N = len(dimensionvec)

    # Compute strides for flattening coordinates (row-major order)
    strides = [1] * N
    for i in range(N - 2, -1, -1):
        strides[i] = strides[i + 1] * dimensionvec[i + 1]

    node_from = []
    node_to = []
    for coord in itertools.product(*(range(sz) for sz in dimensionvec)):
        idx = sum(c * s for c, s in zip(coord, strides))
        for dim in range(N):
            neigh_val = (coord[dim] + 1) % dimensionvec[dim]
            # skip the wrap-around case if we're non-periodic
            if neigh_val == 0 and not periodic:
                continue

            neigh = list(coord)
            neigh[dim] = neigh_val
            jdx = sum(n * s for n, s in zip(neigh, strides))

            node_from.append(idx)
            node_to.append(jdx)

    return node_from, node_to


def random_network(nodesAndConnectivity):
    """
    Generate a random connected network of N nodes and average connectivity C
    with minimum degree >= 2, by sampling E edges and then repairing
    violations via guided additions & safe removals.
    """
    num_nodes, avg_connectivity = nodesAndConnectivity
    if avg_connectivity < 2:
        raise ValueError("Average connectivity must be at least 2.")
    
    total_edges = int(round(num_nodes * avg_connectivity / 2))
    max_edges = num_nodes*(num_nodes-1)//2
    if total_edges > max_edges:
        raise ValueError("Can't make this graph: too many edges requested.")
    
    all_possible = list(itertools.combinations(range(num_nodes), 2))
    edges = set(random.sample(all_possible, total_edges))
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    def remove_safe_edge():
        # compute current bridges
        bridge_set = set(bridges(G))
        safe = [
            e for e in G.edges()
            if e not in bridge_set
               and G.degree[e[0]] > 2
               and G.degree[e[1]] > 2
        ]
        if not safe:
            return
        u, v = random.choice(safe)
        G.remove_edge(u, v)
        edges.remove((min(u, v), max(u, v)))
    
    # repair connectivity
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        comp1 = random.choice(comps)
        comp2 = random.choice([c for c in comps if c is not comp1])
        u = random.choice(list(comp1))
        v = random.choice(list(comp2))
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        G.add_edge(u, v)
        edges.add(e)
        remove_safe_edge()
    
    # repair low‐degree nodes
    low = [n for n, d in G.degree() if d < 2]
    while low:
        u = random.choice(low)
        candidates = set(range(num_nodes)) - {u} - set(G.neighbors(u))
        if not candidates:
            raise RuntimeError(f"Cannot fix degree for node {u}")
        v = random.choice(list(candidates))
        e = (min(u, v), max(u, v))
        G.add_edge(u, v)
        edges.add(e)
        remove_safe_edge()
        low = [n for n, d in G.degree() if d < 2]
    
    node_from, node_to = zip(*edges)
    return list(node_from), list(node_to)

def dense_network_periodic(layer_sizes):
    return dense_network(layer_sizes,periodic=True)

def dense_network(layer_sizes,periodic=False):
    """ Generate a densely connected layered network.
    Each node in layer i is connected to every node in layer i-1 and layer i+1.  """
    
    # Compute starting index of each layer
    offsets = []
    total = 0
    for size in layer_sizes:
        offsets.append(total)
        total += size

    node_from = []
    node_to = []

    # For each adjacent layer pair, connect all nodes between them
    num_layers = len(layer_sizes)
    for layer in range(num_layers - 1+periodic):
        start_u = offsets[layer]
        end_u = offsets[layer] + layer_sizes[layer]
        start_v = offsets[(layer+1)%len(layer_sizes)]
        end_v = offsets[(layer+1)%len(layer_sizes)] + layer_sizes[(layer+1)%len(layer_sizes)]
        for u in range(start_u, end_u):
            for v in range(start_v, end_v):
                # Undirected edge: record once
                node_from.append(u)
                node_to.append(v)
    return node_from, node_to

  