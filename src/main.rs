use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Read, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
// Added for canonical labeling with colors (nauty via graph-canon)
use petgraph::{Graph, Undirected};
use petgraph::graphmap::UnGraphMap;
use graph_canon::canon::CanonLabeling;
use graph_enumeration::check_statement_t_6;


// ---- Parameters (equivalent to Python globals) ---------------------------------
const A_DEGS: [i32; 4] = [4, 4, 3, 3];
const INTERNAL_EDGES: usize = 7;

// ---- Types ---------------------------------------------------------------------
#[inline]
fn popcount_u32(x: u32) -> u32 { x.count_ones() }

#[derive(Clone, Debug)]
struct CanonPair {
    edges: Vec<(usize, usize)>,
    colors: Vec<u32>,
}

// JSON input/output shape: Vec<Graph>, Graph = Vec<Edge>, Edge = [u, v]
#[derive(Serialize, Deserialize)]
struct EdgePair(pub usize, pub usize);

// ---- Utilities -----------------------------------------------------------------
// 必ず小さい方が前に来るように正規化
#[inline]
fn norm_edge(u: usize, v: usize) -> (usize, usize) { if u < v { (u, v) } else { (v, u) } }
// 集合化。重複を許さない。
fn edges_vec_to_set(edges: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    let mut s: HashSet<(usize, usize)> = HashSet::with_capacity(edges.len());
    for &(u, v) in edges { s.insert(norm_edge(u, v)); }
    s
}
// 辺集合に含まれているのかを正規化して確認。
fn set_contains_edge(s: &HashSet<(usize, usize)>, u: usize, v: usize) -> bool {
    s.contains(&norm_edge(u, v))
}

// ---- WL refinement used for orbit reps & canonicalization ----------------------

/// Compute WL-refined color IDs and adjacency lists for vertices [0..used).
fn wl_refine(
    used: usize,
    edges_set: &HashSet<(usize, usize)>,
    colors_sig: &[u32],
    deg: &[i32],
) -> Vec<usize> {
    // Build adjacency
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); used];
    for &(u, v) in edges_set.iter() {
        if u < used && v < used {
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    // Initial partition IDs keyed by (color, deg)
    let mut color_id: BTreeMap<(u32, i32), usize> = BTreeMap::new();
    let mut ids: Vec<usize> = Vec::with_capacity(used);
    for v in 0..used {
        let key = (colors_sig[v], deg[v]);
        let len_now = color_id.len();
        let id = *color_id.entry(key).or_insert(len_now);
        ids.push(id);
    }

    // Refine until stable
    let mut changed = true;
    while changed {
        changed = false;
        let mut palette: BTreeMap<(u32, i32, Vec<(usize, i32)>), usize> = BTreeMap::new();
        let mut new_ids = vec![0usize; used];

        for v in 0..used {
            // histogram of neighbor ids
            let mut hist: BTreeMap<usize, i32> = BTreeMap::new();
            for &w in &adj[v] { *hist.entry(ids[w]).or_insert(0) += 1; }
            let hist_vec: Vec<(usize, i32)> = hist.into_iter().collect();
            let key = (colors_sig[v], deg[v], hist_vec);
            let len_now = palette.len();
            let nid = *palette.entry(key).or_insert(len_now);

            new_ids[v] = nid;
        }
        if new_ids != ids { ids = new_ids; changed = true; }
    }

    ids
}

/// WL-based canonicalization: returns canonically ordered edges & colors.
fn canon_label_bb_with_markers(
    used: usize,
    edges: &[(usize, usize)],
    colors: &[u32],
) -> CanonLabeling {
    // B 頂点 0..used-1 を作成
    let mut g: Graph<(), (), Undirected> = Graph::with_capacity(used, edges.len());
    let b_nodes: Vec<_> = (0..used).map(|_| g.add_node(())).collect();
    for &(u, v) in edges.iter().filter(|&&(u, v)| u < used && v < used) {
        g.add_edge(b_nodes[u], b_nodes[v], ());
    }
    // 色ごとのマーカー頂点を作成し、各頂点を自色マーカーに接続
    let mut marker: BTreeMap<u32, _> = BTreeMap::new();
    for &c in colors.iter().take(used) {
        marker.entry(c).or_insert_with(|| g.add_node(()));
    }
    for i in 0..used {
        let m = marker[&colors[i]];
        g.add_edge(b_nodes[i], m, ());
    }
    CanonLabeling::new(&g)
}

// ---- Build signature pool from bipartite A->B edges ---------------------------

fn build_sig_pool_from_bipartite(
    k_left: usize,
    edges_ab: &[(usize, usize)],
) -> HashMap<u32, usize> {
    // B id -> signature
    let mut sig_of_b: HashMap<usize, u32> = HashMap::new();
    for &(a, b) in edges_ab {
        assert!(a < k_left, "A-side index {a} is out of range 0..{}", k_left - 1);
        let entry = sig_of_b.entry(b).or_insert(0);
        *entry |= 1u32 << (a as u32);
    }

    // Count signatures
    let mut sig_pool: HashMap<u32, usize> = HashMap::new();
    for &sig in sig_of_b.values() { *sig_pool.entry(sig).or_insert(0) += 1; }
    sig_pool
}

// ---- Enumeration with orbit pruning -------------------------------------------

#[derive(Debug, Eq)]
struct MemoKey {
    label: CanonLabeling,
    pool_key: Vec<(u32, usize)>,
}
impl PartialEq for MemoKey {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.pool_key == other.pool_key
    }
}
impl Hash for MemoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        for &(sig, cnt) in &self.pool_key { sig.hash(state); cnt.hash(state); }
    }
}

fn enumerate_union_internal_graphs_orbits_unified(
    x: usize,                  // number of internal B-B edges
    sig_pool: &HashMap<u32, usize>,
    extra0: Option<usize>,     // additional 0000 count (default: 2x) — absorbed into supply
    maxdeg_total: i32,         // per-B degree cap (internal degree + A-connections)
) -> Vec<CanonPair> {
    let zero: u32 = 0;
    let max_slots: usize = 2 * x;
    let extra0_final = extra0.unwrap_or(2 * x);

    // 1) unify supply: signature -> count; include extra0 into signature 0
    let mut supply: HashMap<u32, usize> = sig_pool.clone();
    *supply.entry(zero).or_insert(0) += extra0_final;

    // 2) results & memo per level
    // results labels を使って重複を排除し、results を返す。これはそのまま JSON 出力に使う。
    let mut results_labels: HashSet<CanonLabeling> = HashSet::new();
    let mut results: Vec<CanonPair> = Vec::new();
    let mut seen_per_level: Vec<HashSet<MemoKey>> = (0..=x).map(|_| HashSet::new()).collect();

    // 3) upper bound on future capacity
    let max_future_cap = |
        used_slots: usize,
        supply: &HashMap<u32, usize>,
    | -> i32 {
        let remain_slots = max_slots.saturating_sub(used_slots);
        if remain_slots == 0 { return 0; }
        let mut caps_multiset: Vec<i32> = Vec::new();
        for (&sig, &cnt) in supply.iter() {
            if cnt == 0 { continue; }
            let cap = max(0, maxdeg_total - popcount_u32(sig) as i32);
            if cap > 0 { for _ in 0..cnt { caps_multiset.push(cap); } }
        }
        if caps_multiset.is_empty() { return 0; }
        caps_multiset.sort_unstable_by(|a, b| b.cmp(a)); // desc
        caps_multiset.into_iter().take(remain_slots).sum()
    };

    // 4) WL color refinement for active slots (like Python refine_color_classes)
    let refine_color_classes = |
        used_slots: usize,
        edges_set: &HashSet<(usize, usize)>,
        colors_sig: &[u32],
        deg: &[i32],
    | -> (Vec<usize>, BTreeMap<usize, Vec<usize>>) {
        let ids = wl_refine(used_slots, edges_set, colors_sig, deg);
        let mut classes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for v in 0..used_slots {
            classes.entry(ids[v]).or_default().push(v);
        }
        (ids, classes)
    };

    // 5) pick representative pair from two WL classes
    let pick_rep_pair = |
        ci: &Vec<usize>,
        cj: &Vec<usize>,
        caps: &[i32],
        edges_set: &HashSet<(usize, usize)>,
    | -> Option<(usize, usize)> {
        if std::ptr::eq(ci, cj) {
            let mut cand: Vec<usize> = ci.iter().cloned().filter(|&i| caps[i] > 0).collect();
            cand.sort_unstable();
            for a in 0..cand.len() {
                let u = cand[a];
                for b in (a + 1)..cand.len() {
                    let v = cand[b];
                    if !set_contains_edge(edges_set, u, v) { return Some(norm_edge(u, v)); }
                }
            }
            None
        } else {
            let mut cand_u: Vec<usize> = ci.iter().cloned().filter(|&i| caps[i] > 0).collect();
            cand_u.sort_unstable();
            let mut cand_v: Vec<usize> = cj.iter().cloned().filter(|&j| caps[j] > 0).collect();
            cand_v.sort_unstable();
            for &u in &cand_u {
                for &v in &cand_v {
                    if !set_contains_edge(edges_set, u, v) { return Some(norm_edge(u, v)); }
                }
            }
            None
        }
    };

    // 6) DFS
    fn as_pool_key(supply: &HashMap<u32, usize>) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = supply.iter().map(|(&s, &c)| (s, c)).collect();
        v.sort_unstable_by_key(|x| x.0);
        v
    }

    fn canonical_key(
        used_slots: usize,
        edges: &[(usize, usize)],
        colors: &[u32],
        supply: &HashMap<u32, usize>,
    ) -> MemoKey {
        let label = canon_label_bb_with_markers(used_slots, edges, colors);
        MemoKey { label, pool_key: as_pool_key(supply) }
    }

    fn maybe_push_result(
        used_slots: usize,
        edges: &[(usize, usize)],
        colors: &[u32],
        results_labels: &mut HashSet<CanonLabeling>,
        results: &mut Vec<CanonPair>,
    ) {
        let label = canon_label_bb_with_markers(used_slots, edges, colors);
        if results_labels.insert(label) {
            // 生の (edges, colors) をそのまま保持（正準表示は不要）
            results.push(CanonPair { edges: edges.to_vec(), colors: colors.to_vec() });
        }
    }

    // capture by mutable references
    fn dfs(
        x: usize,
        max_slots: usize,
        maxdeg_total: i32,
        edges: &mut Vec<(usize, usize)>,
        colors_sig: &mut Vec<u32>,
        caps: &mut Vec<i32>,
        deg: &mut Vec<i32>,
        used_slots: usize,
        supply: &mut HashMap<u32, usize>,
        edges_used: usize,
        results_labels: &mut HashSet<CanonLabeling>,
        results: &mut Vec<CanonPair>,
        seen_per_level: &mut [HashSet<MemoKey>],
        max_future_cap: &dyn Fn(usize, &HashMap<u32, usize>) -> i32,
        refine_color_classes: &dyn Fn(usize, &HashSet<(usize, usize)>, &[u32], &[i32]) -> (Vec<usize>, BTreeMap<usize, Vec<usize>>),
        pick_rep_pair: &dyn Fn(&Vec<usize>, &Vec<usize>, &[i32], &HashSet<(usize, usize)>) -> Option<(usize, usize)>,
    ) {
        // Upper-bound pruning
        let cap_active: i32 = caps.iter().take(used_slots).sum();
        let cap_total = cap_active + max_future_cap(used_slots, supply);
        if cap_total / 2 < (x as i32 - edges_used as i32) { return; }

        // Level memoization (canonical edges + supply vector)
        let edges_set = edges_vec_to_set(edges);
        let key = canonical_key(used_slots, edges, colors_sig, supply);
        if seen_per_level[edges_used].contains(&key) { return; }
        seen_per_level[edges_used].insert(key);

        // Target reached
        if edges_used == x {
            maybe_push_result(used_slots, edges, colors_sig, results_labels, results);
            return;
        }

        // Orbit reps via WL classes
        let (_ids, classes) = refine_color_classes(used_slots, &edges_set, colors_sig, deg);
        let mut class_ids_sorted: Vec<usize> = classes.keys().cloned().collect();
        class_ids_sorted.sort_unstable();

        // T1: existing × existing (one representative per class pair)
        for (i_idx, &ci) in class_ids_sorted.iter().enumerate() {
            let Ci = classes.get(&ci).unwrap();
            if let Some((u, v)) = pick_rep_pair(Ci, Ci, caps, &edges_set) {
                edges.push((u, v));
                caps[u] -= 1; caps[v] -= 1;
                deg[u] += 1; deg[v] += 1;
                dfs(
                    x, max_slots, maxdeg_total, edges, colors_sig, caps, deg, used_slots, supply,
                    edges_used + 1, results_labels, results, seen_per_level, max_future_cap, refine_color_classes, pick_rep_pair,
                );
                deg[u] -= 1; deg[v] -= 1;
                caps[u] += 1; caps[v] += 1;
                edges.pop();
            }
            for &cj in class_ids_sorted.iter().skip(i_idx + 1) {
                let Cj = classes.get(&cj).unwrap();
                if let Some((u, v)) = pick_rep_pair(Ci, Cj, caps, &edges_set) {
                    edges.push((u, v));
                    caps[u] -= 1; caps[v] -= 1;
                    deg[u] += 1; deg[v] += 1;
                    dfs(
                        x, max_slots, maxdeg_total, edges, colors_sig, caps, deg, used_slots, supply,
                        edges_used + 1, results_labels, results, seen_per_level, max_future_cap, refine_color_classes, pick_rep_pair,
                    );
                    deg[u] -= 1; deg[v] -= 1;
                    caps[u] += 1; caps[v] += 1;
                    edges.pop();
                }
            }
        }

        // T2/T3: spawn new slots if allowed
        if used_slots < max_slots {
            // candidate signatures (cap>0 and in stock)
            let mut cand_sigs: Vec<u32> = supply
                .iter()
                .filter_map(|(&sig, &cnt)| {
                    if cnt > 0 && (maxdeg_total - popcount_u32(sig) as i32) > 0 { Some(sig) } else { None }
                })
                .collect();
            cand_sigs.sort_unstable();

            // T2: existing × new (existing side uses one rep per class)
            let mut reps_existing: Vec<usize> = Vec::new();
            for &ci in &class_ids_sorted {
                if let Some(list) = classes.get(&ci) {
                    if let Some(&v) = list.iter().find(|&&v| caps[v] > 0) { reps_existing.push(v); }
                }
            }

            for &u in &reps_existing {
                for &sig in &cand_sigs {
                    if supply.get(&sig).copied().unwrap_or(0) == 0 { continue; }
                    let v = used_slots; // new index

                    // activate new slot v
                    let cap_v = max(0, maxdeg_total - popcount_u32(sig) as i32);
                    colors_sig[v] = sig; caps[v] = cap_v; deg[v] = 0;
                    *supply.get_mut(&sig).unwrap() -= 1;

                    if caps[u] > 0 && caps[v] > 0 && !set_contains_edge(&edges_set, u, v) {
                        edges.push((u, v));
                        caps[u] -= 1; caps[v] -= 1; deg[u] += 1; deg[v] += 1;
                        dfs(
                            x, max_slots, maxdeg_total, edges, colors_sig, caps, deg, used_slots + 1, supply,
                            edges_used + 1, results_labels, results, seen_per_level, max_future_cap, refine_color_classes, pick_rep_pair,
                        );
                        deg[u] -= 1; deg[v] -= 1; caps[u] += 1; caps[v] += 1; edges.pop();
                    } else {
                        // Even if we don't add edge u-v immediately, we don't spawn v alone here,
                        // since the Python code tries only paired edge additions in this branch.
                        // (Matches the intent of adding exactly one internal edge per DFS level.)
                        dfs(
                            x, max_slots, maxdeg_total, edges, colors_sig, caps, deg, used_slots + 1, supply,
                            edges_used, results_labels, results, seen_per_level, max_future_cap, refine_color_classes, pick_rep_pair,
                        );
                    }

                    // rollback v
                    *supply.get_mut(&sig).unwrap() += 1;
                    // No need to clear colors_sig[v], caps[v], deg[v] — they will be overwritten on next spawn
                }
            }

            // T3: new × new (spawn two)
            if used_slots + 2 <= max_slots {
                for (i_idx, &sig_i) in cand_sigs.iter().enumerate() {
                    if supply.get(&sig_i).copied().unwrap_or(0) == 0 { continue; }
                    let u = used_slots;
                    let cap_u = max(0, maxdeg_total - popcount_u32(sig_i) as i32);
                    colors_sig[u] = sig_i; caps[u] = cap_u; deg[u] = 0;
                    *supply.get_mut(&sig_i).unwrap() -= 1;

                    for &sig_j in cand_sigs.iter().skip(i_idx) {
                        if supply.get(&sig_j).copied().unwrap_or(0) == 0 { continue; }
                        let v = used_slots + 1;
                        let cap_v = max(0, maxdeg_total - popcount_u32(sig_j) as i32);
                        colors_sig[v] = sig_j; caps[v] = cap_v; deg[v] = 0;
                        *supply.get_mut(&sig_j).unwrap() -= 1;

                        if caps[u] > 0 && caps[v] > 0 && !set_contains_edge(&edges_set, u, v) {
                            edges.push((u, v));
                            caps[u] -= 1; caps[v] -= 1; deg[u] += 1; deg[v] += 1;
                            dfs(
                                x, max_slots, maxdeg_total, edges, colors_sig, caps, deg, used_slots + 2, supply,
                                edges_used + 1, results_labels, results, seen_per_level, max_future_cap, refine_color_classes, pick_rep_pair,
                            );
                            deg[u] -= 1; deg[v] -= 1; caps[u] += 1; caps[v] += 1; edges.pop();
                        }

                        // rollback second
                        *supply.get_mut(&sig_j).unwrap() += 1;
                    }

                    // rollback first
                    *supply.get_mut(&sig_i).unwrap() += 1;
                }
            }
        }
    }

    // 7) initialize and start DFS
    let mut colors_sig = vec![0u32; max_slots];
    let mut caps       = vec![0i32; max_slots];
    let mut deg        = vec![0i32; max_slots];
    let mut edges: Vec<(usize, usize)> = Vec::new();

    dfs(
        x, max_slots, maxdeg_total, &mut edges, &mut colors_sig, &mut caps, &mut deg,
        0, &mut supply, 0, &mut results_labels, &mut results, &mut seen_per_level,
        &max_future_cap, &refine_color_classes, &pick_rep_pair,
    );

    results
}

// ---- Assemble full graph in original labels -----------------------------------

fn assemble_full_graph_complete(
    a_labels: &[usize],                 // e.g., [1,2,3,4]
    canon_edges_bb: &[(usize, usize)],
    canon_colors_b: &[u32],
    total_sig_pool: &HashMap<u32, usize>,
    b_ids_sorted: &[usize],
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let m_active = canon_colors_b.len();
    let mut used: HashMap<u32, usize> = HashMap::new();
    for &sig in canon_colors_b { *used.entry(sig).or_insert(0) += 1; }

    let mut all_sigs: BTreeSet<u32> = BTreeSet::new();
    for &s in total_sig_pool.keys() { all_sigs.insert(s); }
    for &s in used.keys() { all_sigs.insert(s); }

    // remain[sig] = max(0, total - used)
    // BTreeMap を使っているので順番通りにイテレートできる
    let mut remain: BTreeMap<u32, usize> = BTreeMap::new();
    for &s in all_sigs.iter() {
        let total = *total_sig_pool.get(&s).unwrap_or(&0);
        let u = *used.get(&s).unwrap_or(&0);
        remain.insert(s, total.saturating_sub(u));
    }

    let m_remain: usize = remain.values().copied().sum();
    let m_total = m_active + m_remain;

    // Extend B label list if needed
    let mut b_ids_extended: Vec<usize> = b_ids_sorted.to_vec();
    if m_total > b_ids_extended.len() {
        let start = if b_ids_extended.is_empty() { 1 } else { b_ids_extended.iter().copied().max().unwrap() + 1 };
        let need = (m_total - b_ids_extended.len()) as usize;
        for i in 0..need { b_ids_extended.push(start + i); }
    }

    // Map temporary indices 0..m_total-1 to real labels
    let mut b_map: HashMap<usize, usize> = HashMap::new();
    for i in 0..m_total { b_map.insert(i, b_ids_extended[i]); }

    let mut edges_full: Vec<(usize, usize)> = Vec::new();

    // A–B for active B
    for (b_idx, &sig) in canon_colors_b.iter().enumerate() {
        let b_real = b_map[&b_idx];
        for (bit, &a_lab) in a_labels.iter().enumerate() {
            if ((sig >> bit) & 1) == 1 {
                let u = a_lab; let v = b_real;
                edges_full.push(if u < v { (u, v) } else { (v, u) });
            }
        }
    }

    // B–B for active B
    for &(u_idx, v_idx) in canon_edges_bb {
        let u_real = b_map[&u_idx];
        let v_real = b_map[&v_idx];
        let (u, v) = if u_real < v_real { (u_real, v_real) } else { (v_real, u_real) };
        edges_full.push((u, v));
    }

    // Remaining (inactive) B: add only A–B edges
    let mut cur = m_active;
    for (&sig, &cnt) in remain.iter() {
        for _ in 0..cnt {
            let b_real = b_map[&cur];
            for (bit, &a_lab) in a_labels.iter().enumerate() {
                if ((sig >> bit) & 1) == 1 {
                    let u = a_lab; let v = b_real;
                    edges_full.push(if u < v { (u, v) } else { (v, u) });
                }
            }
            cur += 1;
        }
    }

    edges_full.sort_unstable();
    edges_full.dedup();

    (edges_full, b_ids_extended)
}

// ---- Global canonicalization (marker gadgets; A by degree class; B mono) ----
fn canon_label_whole_with_markers(
    a_labels: &[usize],
    b_ids_extended: &[usize],
    edges_full_int: &[(usize, usize)],
) -> CanonLabeling {
    // 収録ノードを列挙
    let mut nodes_set: BTreeSet<usize> = BTreeSet::new();
    for &(u, v) in edges_full_int { nodes_set.insert(u); nodes_set.insert(v); }
    let nodes: Vec<usize> = nodes_set.into_iter().collect();
    let n = nodes.len();

    // 実ラベル -> petgraph ノード index
    let mut g: Graph<(), (), Undirected> = Graph::with_capacity(n + 4, edges_full_int.len() + n);
    let mut idx: HashMap<usize, _> = HashMap::new();
    for &v in &nodes {
        let nd = g.add_node(());
        idx.insert(v, nd);
    }
    // エッジを追加
    for &(u, v) in edges_full_int {
        g.add_edge(idx[&u], idx[&v], ());
    }
    // マーカー頂点：A(4), A(3), B 共通
    let a4_marker = g.add_node(());
    let a3_marker = g.add_node(());
    let b_marker  = g.add_node(());

    let b_set: HashSet<usize> = b_ids_extended.iter().copied().collect();
    // A 側：度クラスごとにマーカーへ
    for &a in a_labels {
        if b_set.contains(&a) { continue; }
        let a_idx = (a - 1) as usize;
        let degv = if a_idx < A_DEGS.len() { A_DEGS[a_idx] } else { 0 };
        if let Some(&a_nd) = idx.get(&a) {
            match degv {
                4 => { let _ = g.add_edge(a_nd, a4_marker, ()); }
                3 => { let _ = g.add_edge(a_nd, a3_marker, ()); }
                _ => {}
            }
        } else {
            // ここで状況を出す
            eprintln!("[DEBUG] A vertex {a} not present in edges_full_int. \
                    nodes={:?}, b_set={:?}", nodes, b_set);
            // 必要なら「無ければ追加」も可能：
            // let a_nd = g.add_node(());
            // idx.insert(a, a_nd);
        }
    }
    // B 側
    for &b in b_ids_extended {
        let b_node = if let Some(&nd) = idx.get(&b) {
            nd
        } else {
            // eprintln!("[WARN] B node {b} missing in idx; adding as isolated.");
            let nd = g.add_node(());
            idx.insert(b, nd);
            nd
        };
        let _ = g.add_edge(b_node, b_marker, ());
    }
    CanonLabeling::new(&g)
}

// ---- Pipeline driver -----------------------------------------------------------

#[derive(Debug)]
struct Config {
    in_json_path: PathBuf,
    out_json_path: PathBuf,
    a_labels: Vec<usize>,
    x_internal_edges: usize,
    maxdeg_total: i32,
}

fn run_pipeline(cfg: &Config) -> Result<()> {
    // Read input JSON: Vec<Vec<[u,v]>>
    let mut f = File::open(&cfg.in_json_path)
        .with_context(|| format!("Failed to open input: {:?}", cfg.in_json_path))?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    // JSON の デシリアライズ
    let graphs_in: Vec<Vec<[usize; 2]>> = serde_json::from_str(&s)
        .with_context(|| "Failed to parse input JSON as Vec<Vec<[usize;2]>>")?;
    // 結果用の Vector
    let mut all_out_graphs: Vec<Vec<[usize; 2]>> = Vec::new();
    // 全体同型の去重複は CanonLabeling をキーに
    // TODO: CanonLabeling を使うので OK なのか？？
    let mut seen_global: HashSet<CanonLabeling> = HashSet::new();

    let k_left = cfg.a_labels.len();
    let amax = *cfg.a_labels.iter().max().unwrap_or(&0);

    for edge_list in graphs_in {
        // 1) split A/B
        let mut edges_ab: Vec<(usize, usize)> = Vec::new();
        let mut b_ids: BTreeSet<usize> = BTreeSet::new();
        for e in edge_list {
            let (u, v) = (e[0], e[1]);
            let (a, b) = if u <= amax { (u, v) } else { (v, u) };
            edges_ab.push(((a - 1) as usize, b));
            b_ids.insert(b);
        }
        let b_ids_sorted: Vec<usize> = b_ids.into_iter().collect();

        // 2) signature pool from bipartite
        let sig_pool = build_sig_pool_from_bipartite(k_left, &edges_ab);

        // 3) enumerate B–B structures
        let sols: Vec<CanonPair> = enumerate_union_internal_graphs_orbits_unified(
            cfg.x_internal_edges, &sig_pool, None, cfg.maxdeg_total,
        );

        // 4) assemble, canonicalize whole graph, dedup
        for CanonPair { edges: canon_edges_bb, colors: canon_colors_b } in sols {
            let (edges_full, b_ids_extended) = assemble_full_graph_complete(
                &cfg.a_labels, &canon_edges_bb, &canon_colors_b, &sig_pool, &b_ids_sorted,
            );
            // TODO: 恐らく問題ないが、CanonLabeling をキーにして良いのか？
            let label = canon_label_whole_with_markers(&cfg.a_labels, &b_ids_extended, &edges_full);
            if !seen_global.insert(label) { continue; } // already seen

            // push in original JSON shape
            let mut one_graph: Vec<[usize; 2]> = Vec::with_capacity(edges_full.len());
            for (u, v) in edges_full { one_graph.push([u, v]); }
            all_out_graphs.push(one_graph);
        }
    }

    // write output JSON
    let f = File::create(&cfg.out_json_path)
        .with_context(|| format!("Failed to create output: {:?}", cfg.out_json_path))?;
    serde_json::to_writer_pretty(f, &all_out_graphs).with_context(|| "Failed to write output JSON")?;

    println!("number of output graphs: {}", all_out_graphs.len());
    Ok(())
}

// ---- CLI ----------------------------------------------------------------------

fn parse_args() -> Result<Config> {
    // Minimal ad-hoc parser to avoid extra deps
    let mut in_json_path = None;
    let mut out_json_path = None;
    let mut a_labels: Vec<usize> = vec![1, 2, 3, 4];
    let mut x_internal_edges: usize = INTERNAL_EDGES;
    let mut maxdeg_total: i32 = 4;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => { in_json_path = Some(PathBuf::from(args[i + 1].clone())); i += 2; }
            "--out" => { out_json_path = Some(PathBuf::from(args[i + 1].clone())); i += 2; }
            "--a-labels" => {
                // comma-separated e.g. 1,2,3,4
                let vals = args[i + 1].split(',').filter(|s| !s.is_empty()).map(|s| s.parse::<usize>().unwrap());
                a_labels = vals.collect();
                i += 2;
            }
            "--x" => { x_internal_edges = args[i + 1].parse::<usize>().unwrap(); i += 2; }
            "--maxdeg" => { maxdeg_total = args[i + 1].parse::<i32>().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    let in_json_path = in_json_path.context("--in <path> is required")?;
    let out_json_path = out_json_path.context("--out <path> is required")?;

    Ok(Config { in_json_path, out_json_path, a_labels, x_internal_edges, maxdeg_total })
}

/// =============== ログ ===============

fn append_logs(log_path: &Path, header: &str, lines: &[String]) -> Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open log {}", log_path.display()))?;
    // 見出し（ファイルを跨いで追記されても識別しやすく）
    writeln!(f, "===== {} =====", header)?;
    for line in lines {
        // lines 自体に改行が含まれている想定なので、そのまま書き出し
        f.write_all(line.as_bytes())?;
        if !line.ends_with('\n') {
            writeln!(f)?;
        }
    }
    Ok(())
}

// ===== JSON 形式（「二重リスト：各グラフ = [ [u,v], ... ]」） =====
#[derive(Deserialize)]
struct AllGraphs(Vec<Vec<[usize; 2]>>);

fn load_graphs_from_json(path: &Path) -> Result<Vec<UnGraphMap<usize, ()>>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let AllGraphs(raw_graphs): AllGraphs =
        serde_json::from_reader(reader)
            .with_context(|| format!("failed to parse JSON {}", path.display()))?;
    let mut graphs: Vec<UnGraphMap<usize, ()>> = Vec::with_capacity(raw_graphs.len());
    for edges in raw_graphs {
        let mut g = UnGraphMap::<usize, ()>::new();
        for [mut u, mut v] in edges {
            if u == v {
                continue; // 自己ループは除外（必要なら保持に変更）
            }
            if v < u {
                std::mem::swap(&mut u, &mut v);
            }
            g.add_edge(u, v, ());
        }
        graphs.push(g);
    }
    Ok(graphs)
}

use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::{Result, Context};
use petgraph::graphmap::UnGraphMap;
use graph6::Graph;

/// graph6 形式から読み込み
fn load_graphs_from_graph6(path: &std::path::Path) -> Result<Vec<UnGraphMap<usize, ()>>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut graphs = Vec::new();
    for line in reader.lines() {
        let line = line?; // 1行1グラフ (graph6 の仕様)
        if line.trim().is_empty() {
            continue;
        }
        let g6 = Graph::from_graph6(&line)
            .with_context(|| format!("failed to parse graph6 line: {}", line))?;
        
        // Graph6 → UnGraphMap に変換
        let mut gmap = UnGraphMap::<usize, ()>::new();
        for u in 0..g6.n() {
            for v in (u+1)..g6.n() {
                if g6.has_edge(u, v) {
                    gmap.add_edge(u, v, ());
                }
            }
        }
        graphs.push(gmap);
    }
    Ok(graphs)
}


fn main() -> Result<()> {
    let start = Instant::now();

    const PART_START: usize = 9;
    const PART_END: usize = 71;
    const INTERNAL_EDGES_SET: [usize;1] = [6];
    // const SKIP_PARTS: &[usize] = &[1,2,3,4,5,6,13,15,18,19,22,29,32,33];
    const SKIP_PARTS: &[usize] = &[];
    const WINDOW: usize = 10; // 調整ポイント：並列度×4 の上限になる

    let in_base = Path::new("graphs/json/new_class/foundation/4_4_3_3_0");
    let out_base = Path::new("graphs/json/new_class/complete/4_4_3_3_0");
    let test_base = Path::new("graphs/json/new_class/complete");
    fs::create_dir_all(out_base)?;

    let skip: HashSet<usize> = SKIP_PARTS.iter().copied().collect();
    let parts: Vec<_> = (PART_START..=PART_END).filter(|p| !skip.contains(p)).collect();
    let total = parts.len() * INTERNAL_EDGES_SET.len();
    let counter = Arc::new(AtomicUsize::new(0));

    for window in parts.chunks(WINDOW) {
        // 窓ごとは順に進む
        window.par_iter().for_each(|&part| {
            let in_path = in_base.join(format!("part_{part}.json"));
            if !in_path.exists() {
                eprintln!("[skip] part {part} not found");
                return;
            }

            // 窓内の各 part では x=3,4,5 を並列
            INTERNAL_EDGES_SET.par_iter().for_each(|&x_internal_edges| {
                let in_path = in_base.join(format!("part_{part}.json"));
                let out_path = out_base.join(format!("part_{part}_{x_internal_edges}.json"));
                let log_path = out_base.join(format!("part_{part}_counter_example.log"));
                let test_out_path = test_base.join(format!("4_4_3_3_0_7_test_rs.json"));
                let test_log_path = test_base.join(format!("4_4_3_3_0_7_test_counter_examples.log"));

                // let cfg = Config {
                //     in_json_path: in_path,
                //     out_json_path: out_path,
                //     a_labels: vec![1, 2, 3, 4],
                //     x_internal_edges,
                //     maxdeg_total: 4,
                // };

                // if let Err(e) = run_pipeline(&cfg) {
                //     eprintln!("[err] part {part}, INTERNAL_EDGES={x_internal_edges}: {e}");
                // }

                // 生成した out_path を読み込み、t=6 でチェック
                let mut any_error = false;
                match load_graphs_from_json(&test_out_path) {
                    Ok(graphs) => {
                        let comment = format!("part_{part}_{x_internal_edges}.json");
                        let logs = check_statement_t_6(&graphs, &comment);
                        if !logs.is_empty() {
                            // 3) 例外ログがあればファイルに追記
                            if let Err(e) = append_logs(&test_log_path, &comment, &logs) {
                                eprintln!("!! failed to write log {}: {:#}", test_log_path.display(), e);
                            }
                            any_error = true;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "!! failed to load generated JSON {}: {:#}",
                            out_path.display(),
                            e
                        );
                        any_error = true;
                    }
                }

                let done = counter.fetch_add(1, Ordering::SeqCst) + 1;
                let progress = (done as f64 / total as f64) * 100.0;
                if any_error {
                    println!(
                        "[{done}/{total} | {progress:5.1}%] finished part {part}, edges={x_internal_edges} (with findings)"
                    );
                } else {
                    println!(
                        "[{done}/{total} | {progress:5.1}%] finished part {part}, edges={x_internal_edges}"
                    );
                }
            });
        });
        // ← この for で“次の窓”に進むので、大崩れしない進行順序と高い並列度を両立
    }

    fs::create_dir_all(out_base)
        .with_context(|| format!("create out base dir {}", out_base.display()))?;

    // 入力ディレクトリ配下の *.json を列挙
    let mut json_files = Vec::<PathBuf>::new();
    for entry in WalkDir::new(out_base).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.into_path();
        if p.extension().and_then(|s| s.to_str()) == Some("json") {
            json_files.push(p);
        }
    }
    if json_files.is_empty() {
        bail!("no JSON files under {}", in_base.display());
    }

    let total = json_files.len();
    let counter = Arc::new(AtomicUsize::new(0));

    // チャンクに分けて順次（各チャンクの中は並列）
    for chunk in json_files.chunks(WINDOW) {
        chunk.par_iter().for_each(|in_path| {
            if let Err(e) = process_one(out_base, out_base, in_path) {
                eprintln!("[err] {}: {:#}", in_path.display(), e);
            }
            let done = counter.fetch_add(1, Ordering::SeqCst) + 1;
            let progress = (done as f64 / total as f64) * 100.0;
            println!("[{done}/{total} | {progress:5.1}%] {}", in_path.display());
        });
    }

    println!("all done in {:.3} 秒", start.elapsed().as_secs_f64());
    Ok(())
}
