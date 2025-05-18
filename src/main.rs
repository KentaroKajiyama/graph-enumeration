use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use petgraph::algo::is_isomorphic;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use wl_isomorphism::invariant; // 1-WL hash
use itertools::Itertools;
mod check_statement;

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
    // Shift back to 0-indexed internally
    graph.add_edge(nodes[u - 1], nodes[v - 1], ());
  }
  graph
}

fn from_petgraph(graph: &MyGraph) -> SerializableGraph {
  let edges = graph.edge_indices()
    .map(|e| {
      let (u, v) = graph.edge_endpoints(e).unwrap();
      // Shift to 1-indexed for output
      (u.index() + 1, v.index() + 1)
    })
    .collect();
  SerializableGraph {
    node_count: graph.node_count(),
    edges,
  }
}

fn load_graphs(path: &str) -> Vec<MyGraph> {
  println!("Trying to open file at: {}", path);
  let file = File::open(path).unwrap();
  let reader = BufReader::new(file);
  let graphs: Vec<SerializableGraph> = serde_json::from_reader(reader).unwrap();
  graphs.iter().map(to_petgraph).collect()
}

fn save_graphs(path: &str, graphs: &[MyGraph]) {
  use indicatif::{ProgressBar, ProgressStyle};

  println!("Saving {} graphs to {}", graphs.len(), path);
  let pb = ProgressBar::new(graphs.len() as u64);
  pb.set_style(
    ProgressStyle::with_template(
      "[save] [{elapsed_precise}] {wide_bar:.green/white} {pos}/{len}"
    ).unwrap()
  );

  // Serialize graph-by-graph with progress
  let mut serializable = Vec::with_capacity(graphs.len());
  for g in graphs {
    serializable.push(from_petgraph(g));
    pb.inc(1);
  }
  pb.finish_with_message("Finished saving graphs");

  // Write to file once
  let file = File::create(path).unwrap();
  let writer = BufWriter::new(file);
  serde_json::to_writer_pretty(writer, &serializable).unwrap();
}

fn generate_semitight_k5_pattern_1() -> Vec<MyGraph> {
  let mut base_graph = Graph::<(), (), Undirected>::with_capacity(0, 0);
  let mut base_nodes = HashMap::new();
  for v in 1..=22 {
    base_nodes.insert(v, base_graph.add_node(()));
  }

  base_graph.extend_with_edges(&[
    (base_nodes[&1], base_nodes[&6]),
    (base_nodes[&1], base_nodes[&7]),
    (base_nodes[&1], base_nodes[&8]),
    (base_nodes[&1], base_nodes[&2]),
  ]);

  let mut tmp_graphs = vec![];

  for edges in [
    vec![(2, 9), (2, 10), (2, 11)],
    vec![(2, 6), (2, 9), (2, 10)],
    vec![(2, 6), (2, 7), (2, 9)],
    vec![(2, 6), (2, 7), (2, 8)],
  ] {
    let mut g = base_graph.clone();
    let mut local_nodes = HashMap::new();
    for v in 1..=22 {
      local_nodes.insert(v, NodeIndex::new(base_nodes[&v].index()));
    }
    for (a, b) in edges {
      g.add_edge(local_nodes[&a], local_nodes[&b], ());
    }
    tmp_graphs.push((g, local_nodes));
  }

  let mut tmp_graphs_2 = vec![];
  for (g, local_nodes) in tmp_graphs {
    for combo in combinations(&(6..15).collect::<Vec<_>>(), 4) {
      let mut g2 = g.clone();
      let ln = local_nodes.clone();
      for &u in &combo {
        g2.add_edge(ln[&3], ln[&u], ());
      }
      tmp_graphs_2.push((g2, ln));
    }
  }

  let tmp_graphs_2: Vec<_> = tmp_graphs_2
    .into_iter()
    .map(|(g, _)| g)
    .collect();
  let tmp_graphs_2 = deduplicate_graphs(tmp_graphs_2);

  let mut tmp_graphs_3 = vec![];
  for g in tmp_graphs_2 {
    let mut local_nodes = HashMap::new();
    for i in 0..g.node_count() {
      local_nodes.insert(i + 1, NodeIndex::new(i)); // assume compact relabeling hasn't happened yet
    }

    for combo in combinations(&(6..19).collect::<Vec<_>>(), 4) {
      let mut g2 = g.clone();
      for &u in &combo {
        g2.add_edge(local_nodes[&4], local_nodes[&u], ());
      }
      tmp_graphs_3.push((g2, local_nodes.clone()));
    }
  }

  let tmp_graphs_3: Vec<_> = tmp_graphs_3
    .into_iter()
    .map(|(g, _)| g)
    .collect();
  let tmp_graphs_3 = deduplicate_graphs(tmp_graphs_3);

  let mut tmp_graphs_4 = vec![];
  for g in tmp_graphs_3 {
    let mut local_nodes = HashMap::new();
    for i in 0..g.node_count() {
      local_nodes.insert(i + 1, NodeIndex::new(i)); // same assumption
    }

    for combo in combinations(&(6..23).collect::<Vec<_>>(), 4) {
      if combo.iter().any(|&u| [6, 7, 8, 9].contains(&u) && g.edges(local_nodes[&u]).count() >= 4) {
        continue;
      }
      let mut g2 = g.clone();
      for &u in &combo {
        g2.add_edge(local_nodes[&5], local_nodes[&u], ());
      }
      tmp_graphs_4.push((g2, local_nodes.clone()));
    }
  }

  let pairs: Vec<(MyGraph, HashMap<usize, NodeIndex>)> = tmp_graphs_4;

  // 2) 次数 < 5 のものだけフィルターして、そのまま map へ
  let result: Vec<MyGraph> = pairs
    .into_iter()
    .filter(|(g, _labels)| {
      g.node_indices().all(|n| g.edges(n).count() < 5)
    })
    .map(|(g, _labels)| {
      // タプルからグラフだけ取り出して渡す
      relabel_compact_graph_preserve_first5(&g)
    })
    .collect();
  
  let deduped = deduplicate_graphs(result);
  deduped
}

fn generate_semitight_k5_pattern_2() -> Vec<MyGraph> {
  // 0) Prepare base graph with 22 nodes
  let mut base_graph = Graph::<(), (), Undirected>::with_capacity(22, 0);
  let mut base_nodes = std::collections::HashMap::new();
  for v in 1..=22 {
    base_nodes.insert(v, base_graph.add_node(()));
  }

  // 1) Add the fixed edges from node 1 to 6,7,8
  base_graph.extend_with_edges(&[
    (base_nodes[&1], base_nodes[&6]),
    (base_nodes[&1], base_nodes[&7]),
    (base_nodes[&1], base_nodes[&8]),
  ]);

  // 2) Pattern2: first attach one of the four base‐edge‐sets at node 2
  let base_edge_sets = [
    vec![(2, 9),  (2, 10), (2, 11), (2, 12)],
    vec![(2, 6),  (2, 9),  (2, 10), (2, 11)],
    vec![(2, 6),  (2, 7),  (2, 9),  (2, 10)],
    vec![(2, 6),  (2, 7),  (2, 8),  (2, 9)],
  ];
  let mut tmp_graphs = Vec::new();
  for edges in &base_edge_sets {
    let mut g = base_graph.clone();
    for &(a, b) in edges {
      g.add_edge(base_nodes[&a], base_nodes[&b], ());
    }
    tmp_graphs.push(g);
  }

  // 3) K₅‐node=3 stage: choose 4 among 6..15
  let mut tmp_graphs_2 = Vec::new();
  for g in &tmp_graphs {
    for comb in (6..16).combinations(4) {
      let mut g2 = g.clone();
      for &u in &comb {
        g2.add_edge(base_nodes[&3], base_nodes[&u], ());
      }
      tmp_graphs_2.push(g2);
    }
  }
  let tmp_graphs_2 = deduplicate_graphs(tmp_graphs_2);

  // 4) K₅‐node=4 stage: choose 4 among 6..19
  let mut tmp_graphs_3 = Vec::new();
  for g in tmp_graphs_2 {
    for comb in (6..20).combinations(4) {
      let mut g2 = g.clone();
      for &u in &comb {
        g2.add_edge(base_nodes[&4], base_nodes[&u], ());
      }
      tmp_graphs_3.push(g2);
    }
  }
  let tmp_graphs_3 = deduplicate_graphs(tmp_graphs_3);

  // 5) K₅‐node=5 stage: choose 4 among 6..22, skip if deg≥4 at 6..9
  let mut tmp_graphs_4 = Vec::new();
  for g in tmp_graphs_3 {
    for comb in (6..23).combinations(4) {
      if comb.iter().any(|&u| [6, 7, 8, 9].contains(&u) &&
          g.edges(base_nodes[&u]).count() >= 4)
      {
        continue;
      }
      let mut g2 = g.clone();
      for &u in &comb {
        g2.add_edge(base_nodes[&5], base_nodes[&u], ());
      }
      tmp_graphs_4.push(g2);
    }
  }

  // 6) Filter out any graph with a node of degree ≥5, relabel, then dedupe
  let result: Vec<MyGraph> = tmp_graphs_4
    .into_iter()
    .filter(|g| {
      g.node_indices().all(|n| g.edges(n).count() < 5)
    })
    .map(|g| relabel_compact_graph_preserve_first5(&g))
    .collect();

  let deduped = deduplicate_graphs(result);
  deduped
}

fn combinations<T: Copy>(slice: &[T], k: usize) -> Vec<Vec<T>> {
  let mut result = vec![];
  let mut combo = Vec::with_capacity(k);
  combine(slice, k, 0, &mut combo, &mut result);
  result
}

fn combine<T: Copy>(
  slice: &[T],
  k: usize,
  start: usize,
  combo: &mut Vec<T>,
  result: &mut Vec<Vec<T>>,
) {
  if combo.len() == k {
    result.push(combo.clone());
    return;
  }
  for i in start..slice.len() {
    combo.push(slice[i]);
    combine(slice, k, i + 1, combo, result);
    combo.pop();
  }
}

fn graph_elimination(graphs: Vec<MyGraph>) -> Vec<MyGraph> {
  graphs
    .into_iter()
    .filter(|g| g.node_indices().all(|n| g.edges(n).count() < 5))
    .collect()
}
/// Deduplicate by WL-hash bucket + exact isomorphism.

fn relabel_compact_graph_preserve_first5(
  graph: &Graph<(), (), Undirected>,
) -> Graph<(), (), Undirected> {
  // old->new の対応表
  let mut mapping: HashMap<NodeIndex, NodeIndex> = HashMap::new();
  let mut new_graph = Graph::<(), (), Undirected>::with_capacity(0, 0);

  // Step1: 内部インデックス 0..=4 (意味的には 外部ラベル 1..=5) を固定追加
  for old_ix in 0..5 {
    let old_node = NodeIndex::new(old_ix);
    // グラフにそのノードが存在し（node_count()の範囲内）、かつ孤立でなければ
    if old_ix < graph.node_count() && graph.edges(old_node).count() > 0 {
      let new_node = new_graph.add_node(());
      mapping.insert(old_node, new_node);
    }
  }

  // Step2: 残りのノードを（内部インデックス 5 以降で）孤立でなければ追加
  for old_node in graph.node_indices().filter(|n| n.index() >= 5) {
    if graph.edges(old_node).count() > 0 {
      let new_node = new_graph.add_node(());
      mapping.insert(old_node, new_node);
    }
  }

  // Step3: 辺を貼る
  for edge in graph.edge_indices() {
    let (u, v) = graph.edge_endpoints(edge).unwrap();
    if let (Some(&u_new), Some(&v_new)) = (mapping.get(&u), mapping.get(&v)) {
      new_graph.add_edge(u_new, v_new, ());
    }
  }

  new_graph
}

fn deduplicate_graphs(graphs: Vec<MyGraph>) -> Vec<MyGraph> {
  // 1) Bucket by WL‐hash
  let mut buckets: HashMap<u64, Vec<MyGraph>> = HashMap::new();
  let pb_hash = ProgressBar::new(graphs.len() as u64);
  pb_hash
    .set_style(
      ProgressStyle::with_template("[hash ] {pos}/{len} {elapsed_precise}")
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );

  for g in graphs {
    // Clone `g` to hand it to `invariant`, then still own `g` for the bucket.
    let h = invariant(g.clone());
    buckets.entry(h).or_default().push(g);
    pb_hash.inc(1);
  }
  pb_hash.finish_with_message("✔ Hashing complete");

  // 2) Exact isomorphism filtering within each bucket
  let total_buckets = buckets.len() as u64;
  let pb_iso = ProgressBar::new(total_buckets);
  pb_iso
    .set_style(
      ProgressStyle::with_template("[iso  ] {pos}/{len} {elapsed_precise}")
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );

  let mut uniques = Vec::new();
  let mut total_before = 0;
  for (_hash, bucket) in buckets {
    total_before += bucket.len();
    let mut bucket_uniques = Vec::new();
    for g in bucket {
      if !bucket_uniques.iter().any(|h2| is_isomorphic(&g, h2)) {
        bucket_uniques.push(g);
      }
    }
    uniques.extend(bucket_uniques);
    pb_iso.inc(1);
  }
  pb_iso.finish_with_message("✔ Isomorphism filtering complete");

  let removed = total_before - uniques.len();
  println!(
    "Deduplication complete: {} → {} ({} removed)",
    total_before,
    uniques.len(),
    removed
  );

  uniques
}

fn dedup_with_labels(
  pairs: Vec<(MyGraph, HashMap<usize, NodeIndex>)>,
) -> Vec<(MyGraph, HashMap<usize, NodeIndex>)> {
  // 1) WL‐ハッシュでバケツ分け
  let mut buckets: HashMap<u64, Vec<(MyGraph, HashMap<usize, NodeIndex>)>> = HashMap::new();
  
  let pb_hash = ProgressBar::new(pairs.len() as u64);
  pb_hash.set_style(
    ProgressStyle::with_template("[hash ] {pos}/{len} {elapsed_precise}")
      .unwrap()
      .progress_chars("█▉▊▋▌▍▎▏  "),
  );
  
  for (g, labels) in pairs {
    let h = invariant(g.clone());
    buckets.entry(h).or_default().push((g, labels));
    pb_hash.inc(1);
  }
  pb_hash.finish_with_message("✔ Hashing complete");
  
  // 2) Exact isomorphism filtering within each bucket
  let total_buckets = buckets.len() as u64;
  let pb_iso = ProgressBar::new(total_buckets);
  pb_iso.set_style(
    ProgressStyle::with_template("[iso  ] {pos}/{len} {elapsed_precise}")
      .unwrap()
      .progress_chars("█▉▊▋▌▍▎▏  "),
  );
  
  let mut uniques = Vec::new();
  let mut total_before = 0;
  for (_h, bucket) in buckets {
    total_before += bucket.len();
    let mut bucket_uniques = Vec::new();
    for (g, labels) in bucket {
      if !bucket_uniques.iter().any(|(g2, _)| is_isomorphic(&g, g2)) {
        bucket_uniques.push((g, labels));
      }
    }
    uniques.extend(bucket_uniques);
    pb_iso.inc(1);
  }
  pb_iso.finish_with_message("✔ Isomorphism filtering complete");
  
  let removed = total_before - uniques.len();
  println!(
    "Deduplication complete: {} → {} ({} removed)",
    total_before,
    uniques.len(),
    removed
  );
  
  uniques
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

fn save_split_graphs(path_prefix: &str, graphs: &[MyGraph], num_parts: usize) {
  use indicatif::{ProgressBar, ProgressStyle};

  let total = graphs.len();
  let chunk_size = (total + num_parts - 1) / num_parts; // ceiling division

  let pb = ProgressBar::new(num_parts as u64);
  pb.set_style(
    ProgressStyle::with_template("[split] {pos}/{len} {elapsed_precise}")
      .unwrap()
      .progress_chars("█▉▊▋▌▍▎▏  "),
  );

  for i in 0..num_parts {
    let start = i * chunk_size;
    let end = usize::min(start + chunk_size, total);
    let chunk = &graphs[start..end];
    let filename = format!("{}part_{}.json", path_prefix, i + 1);
    save_graphs(&filename, chunk);
    pb.inc(1);
  }

  pb.finish_with_message("✔ Split saving complete");
}

// Pattern 1:
fn generate_plus_1_node_or_edge_graphs_pattern_1(input_path: &str, output_path: &str) {
  let pattern_graphs = load_graphs(input_path);
  let mut result = Vec::new();

  // 🟦 Combined Progress Bar: +1 node + +1 edge
  let pb = ProgressBar::new((2 * pattern_graphs.len()) as u64);
  pb.set_style(
    ProgressStyle::with_template("[+1 step] [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len}")
      .unwrap()
  );

  // ➕ +1 node case
  for graph in &pattern_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
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

  // ➕ +1 edge case
  for graph in &pattern_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in (v+1)..graph.node_count() {
        if graph.edges(NodeIndex::new(w)).count() == 4 {
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

fn generate_plus_2_node_or_edge_graphs_pattern_1(input_path: &str, output_path: &str) {
  let plus_1_edge_graphs = load_graphs(input_path);
  let mut result = Vec::new();

  // ✅ Unified progress bar for +2 node and +2 edge
  let total = 2 * plus_1_edge_graphs.len() as u64;
  let pb = ProgressBar::new(total);
  pb.set_style(ProgressStyle::with_template(
    "[+2 step] [{elapsed_precise}] {wide_bar:.yellow/black} {pos}/{len}"
  ).unwrap());

  // ➕ +2 node case
  for graph in &plus_1_edge_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      let mut g = graph.clone();
      let new_node = g.add_node(());
      g.add_edge(NodeIndex::new(v), new_node, ());
      result.push(g);
    }
  }

  // ➕ +2 edge case
  for graph in &plus_1_edge_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in (v+1)..graph.node_count() {
        if graph.edges(NodeIndex::new(w)).count() == 4 {
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

  pb.finish_with_message("Finished +2 node/edge steps");

  let plus_2_edge_graphs = deduplicate_graphs(result);
  save_graphs(output_path, &plus_2_edge_graphs);
}

fn generate_plus_1_node_or_edge_graphs_pattern_2(input_path: &str, output_path: &str) {
  let pattern_graphs = load_graphs(input_path);
  let mut result: Vec<MyGraph> = Vec::new();

  // 🟦 Combined Progress Bar: +1 node + +1 edge
  let pb = ProgressBar::new((2 * pattern_graphs.len()) as u64);
  pb.set_style(
    ProgressStyle::with_template("[+1 step] [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len}")
      .unwrap()
  );

  // ➕ +1 node case
  for graph in &pattern_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
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

  // ➕ +1 edge case
  for graph in &pattern_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in (v+1)..graph.node_count() {
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

  pb.finish_with_message("✔ Finished +1 node/edge");

  let plus_1_edge_graphs = deduplicate_graphs(result);
  save_graphs(output_path, &plus_1_edge_graphs);
}

// +2 node or edge
fn generate_plus_2_node_or_edge_graphs_pattern_2(input_path: &str, output_path: &str) {
  let plus_1_edge_graphs = load_graphs(input_path);
  let mut result: Vec<MyGraph> = Vec::new();

  // ✅ Unified progress bar for +2 node and +2 edge
  let total = 2 * plus_1_edge_graphs.len() as u64;
  let pb = ProgressBar::new(total);
  pb.set_style(
    ProgressStyle::with_template("[+2 step] [{elapsed_precise}] {wide_bar:.yellow/black} {pos}/{len}")
      .unwrap()
  );

  // ➕ +2 node case
  for graph in &plus_1_edge_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      let mut g = graph.clone();
      let new_node = g.add_node(());
      g.add_edge(NodeIndex::new(v), new_node, ());
      result.push(g);
    }
  }

  // ➕ +2 edge case
  for graph in &plus_1_edge_graphs {
    pb.inc(1);
    for v in 5..graph.node_count() {
      if graph.edges(NodeIndex::new(v)).count() == 4 {
        continue;
      }
      for w in (v+1)..graph.node_count() {
        if v == w || graph.edges(NodeIndex::new(w)).count() == 4 {
          continue;
        }
        // skip existing edge
        if graph.find_edge(NodeIndex::new(v), NodeIndex::new(w)).is_some() {
          continue;
        }
        let mut g = graph.clone();
        g.add_edge(NodeIndex::new(v), NodeIndex::new(w), ());
        result.push(g);
      }
    }
  }

  pb.finish_with_message("✔ Finished +2 node/edge steps");

  let plus_2_edge_graphs = deduplicate_graphs(result);
  save_graphs(output_path, &plus_2_edge_graphs);
}

fn main() {
  for part in 1..=20 {
    let inputs = [
      format!("graphs/semitight_K_5_pattern_1/part_{}.json", part),
      format!("graphs/semitight_K_5_pattern_1_plus_1/part_{}.json", part),
      format!("graphs/semitight_K_5_pattern_1_plus_2/part_{}.json", part),
      format!("graphs/semitight_K_5_pattern_2/part_{}.json", part),
      format!("graphs/semitight_K_5_pattern_2_plus_1/part_{}.json", part),
      format!("graphs/semitight_K_5_pattern_2_plus_2/part_{}.json", part),
    ];

    for path in &inputs {
      // If load_graphs takes &str: use path.as_str()
      // If it’s AsRef<Path>, you can pass path directly
      let graphs = load_graphs(path);
      check_statement::check_statement_t_6(&graphs, 6);
      println!("Checked statement for {}", path);
    }
  }
}


