use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use petgraph::visit::{EdgeRef};
use petgraph::algo::is_bipartite_undirected;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::{DMatrix, SVD};
use rand::{SeedableRng, rngs::StdRng, rngs::OsRng, Rng};
use rand_distr::StandardNormal;
use std::collections::{HashMap, HashSet, VecDeque};

/// Check statement for symmetric tensor matroid of size t=6
pub fn check_statement_t_6(graphs: &[Graph<(), (), Undirected>], t: usize) {
  let pb = ProgressBar::new(graphs.len() as u64);
  pb.set_style(
    ProgressStyle::with_template("[check] {pos}/{len} {elapsed_precise} – edge size = {wide_msg}")
      .unwrap()
      .progress_chars("█▉▊▋▌▍▎▏  "),
  );

  for G in graphs.iter() {
    pb.inc(1);
    pb.set_message(format!("t = {}", t));

    // 1) Skip if any vertex degree > 5
    if G.node_indices().any(|v| G.edges(v).count() > 5) {
      continue;
    }

    // 2) Compute rank and compare with edge count
    let rank = symmetric_tensor_matroid_rank_graph(G, t, Some(42));
    let m = G.edge_count();
    if rank == 21 || m == rank {
      continue;
    }

    // 3) Find a circuit and build complete graph on same vertex set
    let circuit = match find_one_circuit_in_symmetric_tensor_matroid(G, t, Some(42)) {
      Some(c) => c,
      None => {
        println!("there is no circuit");
        continue;           // skip to the next G in graphs.iter()
      }
    };
    let n = G.node_count();
    let mut complete = Graph::<(), (), Undirected>::with_capacity(n, 0);
    // add nodes 0..n-1
    for _ in 0..n {
      complete.add_node(());
    }
    // add all possible edges
    for i in 0..n {
      for j in (i+1)..n {
        complete.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
      }
    }

    let closure = compute_closure_from_an_instance(&complete, &circuit, t, Some(42));

    // 4) Check membership in C_n^t
    if !is_in_c_n_t(&closure, t) {
      println!("This may be a counter example");
      println!("G's rank: {}", rank);
      println!("G nodes: {:?}", G.node_indices().collect::<Vec<_>>());
      println!("G edges: {:?}", G.edge_references().map(|e| (e.source(), e.target())).collect::<Vec<_>>());
      println!("closure nodes: {:?}", closure.node_indices().collect::<Vec<_>>());
      println!("closure edges: {:?}", closure.edge_references().map(|e| (e.source(), e.target())).collect::<Vec<_>>());
    }
  }

  pb.finish_with_message("✔ check complete");
}

/// Recursive membership test for C_n^t
pub fn is_in_c_n_t(closure: &Graph<(), (), Undirected>, t: usize) -> bool {
  // 1) Remove isolated nodes
  let mut cleaned = closure.clone();
  let isolated: Vec<_> = cleaned
    .node_indices()
    .filter(|&v| closure.edges(v).count() == 0)
    .collect();
  for v in isolated {
    cleaned.remove_node(v);
  }

  let n = cleaned.node_count();
  // 2) If t == 1: check empty or complete
  if t == 1 {
    let m = cleaned.edge_count();
    return m == n*(n-1)/2 || m == 0;
  }

  // 3) If there's a vertex of degree n-1, remove it recursively
  if let Some(v) = cleaned
    .node_indices()
    .find(|&v| cleaned.edges(v).count() == n-1)
  {
    let mut reduced = cleaned.clone();
    reduced.remove_node(v);
    return is_in_c_n_t(&reduced, t-1);
  }

  // 4) If no edges: OK
  if cleaned.edge_count() == 0 {
    return true;
  }

  // 5) Bipartite checks (size-based criteria)
  if let Some(start) = cleaned.node_indices().next() {
    if is_bipartite_undirected(&cleaned, start) {
      // TODO: extract sets U,V via your chosen method
      let (u_set, v_set) = match bipartite_sets(&cleaned) {
        Some((u,v)) => (u, v),
        None => return false,
      };
      let m = cleaned.edge_count();
      match t {
        6 => {
          let us = u_set.len(); let vs = v_set.len();
          return match (us, vs) {
            (3,5) | (5,3) => m == 15,
            (4,5) | (5,4) => m == 20,
            (4,4)        => m == 16,
            (5,5)        => m == 25,
            _            => false,
          };
        }
        5 => {
          let us = u_set.len(); let vs = v_set.len();
          return match (us, vs) {
            (3,4) | (4,3) => m == 12,
            (4,4)        => m == 16,
            _            => false,
          };
        }
        4 => {
          let us = u_set.len(); let vs = v_set.len();
          return (us, vs) == (3,3) && m == 9;
        }
        _ => {}
      }
    }
  }
  // else: counterexample
  println!("This may be a counter example");
  return false;
}


/// Compute rank of symmetric-tensor matroid for graph G in R^t
pub fn symmetric_tensor_matroid_rank_graph(
  graph: &Graph<(), (), Undirected>,
  t: usize,
  seed: Option<u64>,
) -> usize {
  // RNG setup
  let mut rng = if let Some(s) = seed {
    StdRng::seed_from_u64(s)
  } else {
    let mut os_rng = OsRng;
    StdRng::from_rng(&mut os_rng).unwrap()
  };

  // Map nodes to indices 0..n-1
  let node_list: Vec<NodeIndex> = graph.node_indices().collect();
  let n = node_list.len();
  let mut node_index = std::collections::HashMap::new();
  for (i, &v) in node_list.iter().enumerate() {
    node_index.insert(v, i);
  }

  // Random embedding p: n x t
  let mut p = DMatrix::<f64>::zeros(n, t);
  for i in 0..n {
    for j in 0..t {
      p[(i,j)] = rng.sample(StandardNormal);
    }
  }

  // Collect edges
  let edges: Vec<(NodeIndex, NodeIndex)> =
    graph.edge_references().map(|e| (e.source(), e.target())).collect();
  let m = edges.len();

  // Dimension of Sym^2(R^t)
  let d = t*(t+1)/2;
  let mut M = DMatrix::<f64>::zeros(d, m);

  // Flatten upper triangle helper
  let mut flat = Vec::with_capacity(d);
  for (col_idx, &(u,v)) in edges.iter().enumerate() {
    let i = node_index[&u];
    let j = node_index[&v];
    // Outer products
    let vi = p.row(i).transpose();   // t x 1
    let vj = p.row(j).transpose();   // t x 1
    let outer_uv = &vi * vj.transpose();
    let outer_vu = &vj * vi.transpose();
    let sym_uv = outer_uv + outer_vu;
    flat.clear();
    for r in 0..t {
      for c in r..t {
        flat.push(sym_uv[(r,c)]);
      }
    }
    for row in 0..d {
      M[(row, col_idx)] = flat[row];
    }
  }

  // Compute rank via SVD
  let svd = SVD::new(M, false, false);
  let eps = 1e-8;
  svd.singular_values.iter().filter(|&&x| x > eps).count()
}

/// Find one circuit in symmetric-tensor matroid, return subgraph of G
pub fn find_one_circuit_in_symmetric_tensor_matroid(
  graph: &Graph<(), (), Undirected>,
  t: usize,
  seed: Option<u64>,
) -> Option<Graph<(), (), Undirected>> {
  // RNG
  let mut rng = if let Some(s) = seed {
    rand::rngs::StdRng::seed_from_u64(s)
  } else {
    rand::rngs::StdRng::from_entropy()
  };

  // Node mapping
  let node_list: Vec<NodeIndex> = graph.node_indices().collect();
  let n = node_list.len();
  let mut idx_map = std::collections::HashMap::new();
  for (i, &v) in node_list.iter().enumerate() {
    idx_map.insert(v, i);
  }

  // Embedding p
  let mut p = DMatrix::<f64>::zeros(n, t);
  for i in 0..n { for j in 0..t { p[(i,j)] = rng.sample(StandardNormal); }}

  let edges: Vec<(NodeIndex, NodeIndex)> =
    graph.edge_references().map(|e| (e.source(), e.target())).collect();
  let m = edges.len();
  let d = t*(t+1)/2;

  // Build columns
  let mut cols: Vec<DMatrix<f64>> = Vec::with_capacity(m);
  let mut flat = Vec::with_capacity(d);
  for &(u,v) in &edges {
    let i = idx_map[&u];
    let j = idx_map[&v];
    let vi = p.row(i).transpose();
    let vj = p.row(j).transpose();
    let sym_uv = &vi * vj.transpose() + &vj * vi.transpose();
    flat.clear();
    for r in 0..t { for c in r..t { flat.push(sym_uv[(r,c)]); }}
    cols.push(DMatrix::from_column_slice(d, 1, &flat));
  }

  let mut B = DMatrix::<f64>::zeros(d, 0);
  let mut basis: Vec<usize> = Vec::new();
  let eps = 1e-8;

  for (eidx, col) in cols.iter().enumerate() {
    // trial = [B | col]
    let trial = DMatrix::from_columns(
      &B.column_iter()
        .chain(std::iter::once(col.column(0)))
        .collect::<Vec<_>>()
    );

    let old_rank = SVD::new(B.clone(), false, false)
      .singular_values.iter().filter(|&&x| x>eps).count();
    let new_rank = SVD::new(trial.clone(), false, false)
      .singular_values.iter().filter(|&&x| x>eps).count();
    if new_rank == old_rank {
      // dependent → extract circuit
      let svd = SVD::new(trial, false, true);
      let v_t = svd.v_t.unwrap();
      let dep = v_t.row(v_t.nrows()-1);
      // collect indices with nonzero coeff
      let mut circ_idx = Vec::new();
      for (i, &bc) in basis.iter().enumerate() {
        if dep[i].abs()>eps { circ_idx.push(bc); }
      }
      circ_idx.push(eidx);
      // build subgraph
      let mut cg = Graph::<(), (), Undirected>::with_capacity(n, 0);
      for _ in 0..n { cg.add_node(()); }
      for &ci in &circ_idx {
        let (u,v) = edges[ci];
        cg.add_edge(u, v, ());
      }
      return Some(cg);
    } else {
      B = trial;
      basis.push(eidx);
    }
  }
  None
}

/// Compute closure: add edges that don't increase rank
pub fn compute_closure_from_an_instance(
  graph: &Graph<(), (), Undirected>,
  instance: &Graph<(), (), Undirected>,
  t: usize,
  seed: Option<u64>,
) -> Graph<(), (), Undirected> {
  let eps = 1e-8;
  // RNG & embedding
  let mut rng = if let Some(s)=seed {
    rand::rngs::StdRng::seed_from_u64(s)
  } else { rand::rngs::StdRng::from_entropy() };
  let node_list: Vec<NodeIndex> = graph.node_indices().collect();
  let n = node_list.len();
  let mut idx_map = std::collections::HashMap::new();
  for (i,&v) in node_list.iter().enumerate() { idx_map.insert(v,i); }
  let mut p = DMatrix::<f64>::zeros(n,t);
  for i in 0..n { for j in 0..t { p[(i,j)] = rng.sample(StandardNormal); }}

  // Build M for instance edges
  let inst_edges: Vec<(NodeIndex,NodeIndex)> = instance
    .edge_references().map(|e|(e.source(), e.target())).collect();
  let m0 = inst_edges.len();
  let d = t*(t+1)/2;
  let mut M = DMatrix::<f64>::zeros(d, m0);
  let mut flat = Vec::with_capacity(d);
  for (ci,&(u,v)) in inst_edges.iter().enumerate() {
    let i = idx_map[&u]; let j = idx_map[&v];
    let vi=p.row(i).transpose(); let vj=p.row(j).transpose();
    let sym_uv = &vi*vj.transpose() + &vj*vi.transpose();
    flat.clear(); for r in 0..t{ for c in r..t{ flat.push(sym_uv[(r,c)]); }}
    for r in 0..d { M[(r,ci)] = flat[r]; }
  }

  // Outside edges
  let all: Vec<_> = graph.edge_references()
    .map(|e|(e.source(), e.target())).collect();
  let inst_set: std::collections::HashSet<_> = inst_edges.iter().cloned().collect();
  let mut closure_edges = inst_edges.clone();
  for &(u,v) in all.iter() {
    if inst_set.contains(&(u,v)) || inst_set.contains(&(v,u)) { continue; }
    // form new column
    let i=idx_map[&u]; let j=idx_map[&v];
    let vi=p.row(i).transpose(); let vj=p.row(j).transpose();
    let sym_uv=&vi*vj.transpose()+&vj*vi.transpose();
    flat.clear(); for r in 0..t{ for c in r..t{ flat.push(sym_uv[(r,c)]);} }
    let col = DMatrix::from_column_slice(d,1,&flat);
    let trial = DMatrix::from_columns(
      &M.column_iter()
        .chain(std::iter::once(col.column(0)))
        .collect::<Vec<_>>()
    );
    let old_rank = SVD::new(M.clone(), false, false)
      .singular_values.iter().filter(|&&x| x>eps).count();
    let new_rank = SVD::new(trial.clone(), false, false)
      .singular_values.iter().filter(|&&x| x>eps).count();
    if new_rank==old_rank {
      M = trial;
      closure_edges.push((u,v));
    }
  }

  // Build closure graph
  let mut cg = Graph::<(), (), Undirected>::with_capacity(n,0);
  for _ in 0..n { cg.add_node(()); }
  for (u,v) in closure_edges { cg.add_edge(u,v,()); }
  cg
}

pub fn bipartite_sets(
  graph: &Graph<(), (), petgraph::Undirected>
) -> Option<(HashSet<NodeIndex>, HashSet<NodeIndex>)> {
  let mut color: HashMap<NodeIndex, bool> = HashMap::new();
  let mut queue: VecDeque<NodeIndex> = VecDeque::new();

  for start in graph.node_indices() {
    if color.contains_key(&start) {
      continue;
    }
    color.insert(start, false);
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
      let u_color = color[&u];
      for v in graph.neighbors(u) {
        if let Some(&v_color) = color.get(&v) {
          if v_color == u_color {
            return None; // Not bipartite
          }
        } else {
          color.insert(v, !u_color);
          queue.push_back(v);
        }
      }
    }
  }

  let mut u_set = HashSet::new();
  let mut v_set = HashSet::new();
  for (node, is_u) in color {
    if is_u {
      u_set.insert(node);
    } else {
      v_set.insert(node);
    }
  }
  Some((u_set, v_set))
}