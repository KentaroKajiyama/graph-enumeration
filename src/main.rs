use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use petgraph::algo::is_isomorphic;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

type MyGraph = Graph<(), (), Undirected>;

#[derive(Serialize, Deserialize)]
struct SerializableGraph {
  node_count: usize,
  edges: Vec<(usize, usize)>,
}

fn to_petgraph(g: &SerializableGraph) -> MyGraph {
  let mut graph = MyGraph::new_undirected();
  let nodes: Vec<NodeIndex> = (0..g.node_count).map(|_| graph.add_node(())).collect();
  for &(u, v) in &g.edges {
    graph.add_edge(nodes[u], nodes[v], ());
  }
  graph
}

fn from_petgraph(graph: &MyGraph) -> SerializableGraph {
  let edges = graph.edge_indices()
    .map(|e| {
      let (u, v) = graph.edge_endpoints(e).unwrap();
      (u.index(), v.index())
    })
    .collect();
  SerializableGraph {
    node_count: graph.node_count(),
    edges,
  }
}

fn load_graphs(path: &str) -> Vec<MyGraph> {
  let file = File::open(path).unwrap();
  let reader = BufReader::new(file);
  let graphs: Vec<SerializableGraph> = serde_json::from_reader(reader).unwrap();
  graphs.iter().map(to_petgraph).collect()
}

fn save_graphs(path: &str, graphs: &[MyGraph]) {
  let serializable: Vec<SerializableGraph> = graphs.iter().map(from_petgraph).collect();
  let file = File::create(path).unwrap();
  let writer = BufWriter::new(file);
  serde_json::to_writer_pretty(writer, &serializable).unwrap();
}

fn degree_sequence_hash(graph: &MyGraph) -> Vec<usize> {
  let mut degrees: Vec<usize> = graph.node_indices().map(|n| graph.edges(n).count()).collect();
  degrees.sort_unstable();
  degrees
}

fn deduplicate_graphs(graphs: Vec<MyGraph>) -> Vec<MyGraph> {
  let mut buckets: HashMap<Vec<usize>, Vec<MyGraph>> = HashMap::new();

  for g in graphs {
    let hash = degree_sequence_hash(&g);
    buckets.entry(hash).or_default().push(g);
  }

  let mut unique_graphs = Vec::new();
  for group in buckets.values() {
    let mut uniques: Vec<MyGraph> = Vec::new();
    for g in group {
      if uniques.iter().all(|h| !is_isomorphic(g, h)) {
        uniques.push(g.clone());
      }
    }
    unique_graphs.extend(uniques);
  }

  unique_graphs
}

fn relabel_compact_graph(graph: &Graph<(), (), Undirected>) -> Graph<(), (), Undirected> {
  let mut mapping = HashMap::new();
  let mut new_graph = Graph::new_undirected();

  // Step 1: Add non-isolated nodes with new indices
  for node in graph.node_indices() {
    if graph.edges(node).count() > 0 {
      let new_node = new_graph.add_node(());
      mapping.insert(node, new_node);
    }
  }

  // Step 2: Add edges using new node indices
  for edge in graph.edge_indices() {
    let (u, v) = graph.edge_endpoints(edge).unwrap();
    if let (Some(&u_new), Some(&v_new)) = (mapping.get(&u), mapping.get(&v)) {
      new_graph.add_edge(u_new, v_new, ());
    }
  }
  new_graph
}

fn generate_plus_1_node_or_edge_graphs(input_path: &str, output_path: &str) {
  let pattern_graphs = load_graphs(input_path);
  let mut result = Vec::new();

  // Progress bar
  let pb = ProgressBar::new(pattern_graphs.len() as u64);
  pb.set_style(ProgressStyle::with_template(
    "[+1 node/edge] [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len}"
  ).unwrap());

  // +1 node case
  for graph in &pattern_graphs {
    pb.inc(1);
    for v in 6..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      let mut g = graph.clone();
      let new_node = g.add_node(());
      g.add_edge(NodeIndex::new(v), new_node, ());
      let g_clean = relabel_compact_graph(&g);
      result.push(g_clean);
    }
  }

  // +1 edge case
  for graph in &pattern_graphs {
    for v in 6..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in 7..graph.node_count() {
        if w == v || graph.edges(NodeIndex::new(w)).count() == 4 {
          continue;
        }
        let mut g = graph.clone();
        g.add_edge(NodeIndex::new(v), NodeIndex::new(w), ());
        let g_clean = relabel_compact_graph(&g);
        result.push(g_clean);
      }
    }
  }

  pb.finish_with_message("Finished +1 node/edge");
  let plus_1_edge_graphs = deduplicate_graphs(result);
  save_graphs(output_path, &plus_1_edge_graphs);
}

use indicatif::{ProgressBar, ProgressStyle};

fn generate_plus_2_node_or_edge_graphs(input_path: &str, output_path: &str) {
  let plus_1_edge_graphs = load_graphs(input_path);
  let mut result = Vec::new();

  // Progress bar for +1 more node
  let pb_node = ProgressBar::new(plus_1_edge_graphs.len() as u64);
  pb_node.set_style(ProgressStyle::with_template(
    "[+2 node] [{elapsed_precise}] {wide_bar:.green/white} {pos}/{len}"
  ).unwrap());

  for graph in &plus_1_edge_graphs {
    pb_node.inc(1);
    for v in 6..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      let mut g = graph.clone();
      let new_node = g.add_node(());
      g.add_edge(NodeIndex::new(v), new_node, ());
      result.push(g);
    }
  }
  pb_node.finish_with_message("Finished +2 node");

  // Progress bar for +1 more edge
  let pb_edge = ProgressBar::new(plus_1_edge_graphs.len() as u64);
  pb_edge.set_style(ProgressStyle::with_template(
    "[+2 edge] [{elapsed_precise}] {wide_bar:.magenta/white} {pos}/{len}"
  ).unwrap());

  for graph in &plus_1_edge_graphs {
    pb_edge.inc(1);
    for v in 6..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in 7..graph.node_count() {
        if v == w || graph.edges(NodeIndex::new(w)).count() == 4 {
          continue;
        }
        if graph.find_edge(NodeIndex::new(v), NodeIndex::new(w)).is_some() {
          continue;
        }
        let mut g = graph.clone();
        g.add_edge(NodeIndex::new(v), NodeIndex::new(w), ());
        result.push(g);
      }
    }
  }
  pb_edge.finish_with_message("Finished +2 edge");

  let plus_2_edge_graphs = deduplicate_graphs(result);
  save_graphs(output_path, &plus_2_edge_graphs);
}


fn main() {
  for i in 1..=20 {
    let input_path = format!("graphs/json/semitight_K_5_pattern_2/part_{}.json", i);
    let output_path = format!("graphs/json/semitight_K_5_pattern_2_plus_1/part_{}.json", i);
    generate_plus_1_node_or_edge_graphs(&input_path, &output_path);
    println!("Finished generating +1 node/edge graphs for part {}", i);
  }
}

