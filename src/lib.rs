use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use petgraph::graphmap::UnGraphMap;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use std::collections::{HashMap, HashSet};



/// =============== 乱数ベクトル生成 ===============

fn random_normal_matrix(n: usize, t: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
    // p[i] は長さ t のベクトル（ノード i に対応）
    let mut rng: StdRng = match seed {
        Some(s) => SeedableRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut p = vec![vec![0.0; t]; n];
    for i in 0..n {
        for j in 0..t {
            let x = StandardNormal.sample(&mut rng);
            p[i][j] = x;
        }
    }
    p
}

/// =============== 行列ユーティリティ（列ベース） ===============
/// このコードでは列ベクトル集合を頻繁に扱うため、列志向のヘルパーを用意。
/// M は「行 = d、列 = m」を仮定して row-major の Vec<Vec<f64>> で表現します。

fn zeros_matrix(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; cols]; rows]
}

fn concat_columns(a: &Vec<Vec<f64>>, col: &[f64]) -> Vec<Vec<f64>> {
    // a: (rows x k), col: (rows), 返り値: (rows x (k+1))
    let rows = a.len();
    let k = if rows == 0 { 0 } else { a[0].len() };
    assert!(rows == col.len() || rows == 0);
    let mut out = if rows == 0 {
        // 特殊ケース: 空行列 + 列
        let mut m = vec![vec![0.0; 1]; col.len()];
        for r in 0..col.len() {
            m[r][0] = col[r];
        }
        return m;
    } else {
        zeros_matrix(rows, k + 1)
    };
    for r in 0..rows {
        for c in 0..k {
            out[r][c] = a[r][c];
        }
        out[r][k] = col[r];
    }
    out
}

fn remove_column(a: &Vec<Vec<f64>>, idx: usize) -> Vec<Vec<f64>> {
    // 返り値: 列 idx を取り除いた行列
    if a.is_empty() {
        return a.clone();
    }
    let rows = a.len();
    let cols = a[0].len();
    assert!(idx < cols);
    let mut out = zeros_matrix(rows, cols - 1);
    for r in 0..rows {
        let mut c_out = 0;
        for c in 0..cols {
            if c == idx {
                continue;
            }
            out[r][c_out] = a[r][c];
            c_out += 1;
        }
    }
    out
}

/// =============== ランク計算（数値ガウス消去） ===============
/// BLAS/LAPACK なしで動くように、部分ピボット付きガウス消去でランクを求めます。
/// 乱数生成で"generic"な列がほとんどなので、実務上十分に安定に動きます。

fn matrix_max_abs(a: &Vec<Vec<f64>>) -> f64 {
    let mut m = 0.0_f64;
    for row in a {
        for &x in row {
            let ax = x.abs();
            if ax > m {
                m = ax;
            }
        }
    }
    m
}

fn matrix_rank(a: &Vec<Vec<f64>>) -> usize {
    let mut m = a.clone();
    if m.is_empty() || m[0].is_empty() {
        return 0;
    }
    let rows = m.len();
    let cols = m[0].len();

    let mut r = 0usize;
    let max_abs = matrix_max_abs(&m);
    let eps = f64::EPSILON;
    let tol = (rows.max(cols) as f64) * eps * max_abs.max(1.0);

    for c in 0..cols {
        // ピボット行を探す
        let mut pivot = r;
        let mut best = 0.0_f64;
        for i in r..rows {
            let val = m[i][c].abs();
            if val > best {
                best = val;
                pivot = i;
            }
        }
        if best <= tol {
            // この列ではピボット無し
            continue;
        }
        // ピボット行を r と入れ替え
        if pivot != r {
            m.swap(pivot, r);
        }

        // 下を消去
        let pivot_val = m[r][c];
        for i in (r + 1)..rows {
            let f = m[i][c] / pivot_val;
            if f.abs() > 0.0 {
                for j in c..cols {
                    m[i][j] -= f * m[r][j];
                }
            }
        }
        r += 1;
        if r == rows {
            break;
        }
    }
    r
}

/// =============== テンソル列の構築 ===============

fn outer(u: &[f64], v: &[f64]) -> Vec<Vec<f64>> {
    let t = u.len();
    assert_eq!(t, v.len());
    let mut m = vec![vec![0.0; t]; t];
    for i in 0..t {
        for j in 0..t {
            m[i][j] = u[i] * v[j];
        }
    }
    m
}

fn add_inplace(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for (ra, rb) in a.iter_mut().zip(b.iter()) {
        for (xa, &xb) in ra.iter_mut().zip(rb.iter()) {
            *xa += xb;
        }
    }
}

fn flatten_upper_tri(sym: &[Vec<f64>]) -> Vec<f64> {
    let t = sym.len();
    let mut out = Vec::with_capacity(t * (t + 1) / 2);
    for r in 0..t {
        for c in r..t {
            out.push(sym[r][c]);
        }
    }
    out
}

/// =============== 安定なエッジ順 ===============

fn stable_edges(g: &UnGraphMap<usize, ()>) -> Vec<(usize, usize)> {
    g.all_edges()
        .map(|(u, v, _)| if u < v { (u, v) } else { (v, u) })
        .sorted()
        .collect()
}

/// =============== ランク関数（Python: symmetric_tensor_matroid_rank_graph） ===============

pub fn symmetric_tensor_matroid_rank_graph(
    g: &UnGraphMap<usize, ()>,
    t: usize,
    seed: Option<u64>,
) -> usize {
    let nodes: Vec<usize> = g.nodes().sorted().collect();
    let n = nodes.len();
    let index: HashMap<usize, usize> =
        nodes.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();

    let p = random_normal_matrix(n, t, seed);

    let edges = stable_edges(g);
    let m_edges = edges.len();
    let d = t * (t + 1) / 2;

    let mut mtx = zeros_matrix(d, m_edges);

    for (col, (u, v)) in edges.iter().enumerate() {
        let iu = *index.get(u).unwrap();
        let iv = *index.get(v).unwrap();
        let out_uv = outer(&p[iu], &p[iv]);
        let out_vu = outer(&p[iv], &p[iu]);
        let mut sym = out_uv;
        add_inplace(&mut sym, &out_vu);
        let col_vec = flatten_upper_tri(&sym);
        for r in 0..d {
            mtx[r][col] = col_vec[r];
        }
    }

    matrix_rank(&mtx)
}

/// =============== サーキット検出（Python: find_one_circuit_in_symmetric_tensor_matroid） ===============

pub fn find_one_circuit_in_symmetric_tensor_matroid(
    g: &UnGraphMap<usize, ()>,
    t: usize,
    seed: Option<u64>,
) -> Option<UnGraphMap<usize, ()>> {
    let nodes: Vec<usize> = g.nodes().sorted().collect();
    let n = nodes.len();
    let index: HashMap<usize, usize> =
        nodes.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
    let p = random_normal_matrix(n, t, seed);

    let edges = stable_edges(g);
    let m = edges.len();
    let d = t * (t + 1) / 2;

    // 列ベクトル（長さ d）を m 本
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(m);
    for &(u, v) in &edges {
        let iu = *index.get(&u).unwrap();
        let iv = *index.get(&v).unwrap();
        let mut sym = outer(&p[iu], &p[iv]);
        let out_vu = outer(&p[iv], &p[iu]);
        add_inplace(&mut sym, &out_vu);
        columns.push(flatten_upper_tri(&sym));
    }

    // 現在の基底行列 B と、その列に対応する「元のエッジ添字」basis_edges
    let mut b = zeros_matrix(d, 0);
    let mut basis_edges: Vec<usize> = Vec::new();

    for eidx in 0..m {
        let trial = concat_columns(&b, &columns[eidx]);
        let old_rank = matrix_rank(&b);
        let new_rank = matrix_rank(&trial);

        if new_rank == old_rank {
            // 依存が起きた。極小従属集合（回路）の推定。
            // 重要：trial は「基底列（|basis_edges| 本） + 新列」の順序。
            // 各基底列を1本ずつ外して new_rank と同じなら、その列は従属関係に寄与している可能性が高い。
            let mut circuit_positions: Vec<usize> = Vec::new();
            for pos in 0..basis_edges.len() {
                let test = remove_column(&trial, pos);
                if matrix_rank(&test) == new_rank {
                    circuit_positions.push(pos);
                }
            }
            // 新列（最後の列）は常にサーキットに含める
            // （Python 版と同様の意図。ただし Python は添字バグがあったため、Rust では列位置で扱う）
            let mut circuit_edges: Vec<(usize, usize)> = circuit_positions
                .into_iter()
                .map(|pos| edges[basis_edges[pos]])
                .collect();
            circuit_edges.push(edges[eidx]);

            let mut sub = UnGraphMap::<usize, ()>::new();
            for &u in &nodes {
                sub.add_node(u);
            }
            for (u, v) in circuit_edges {
                sub.add_edge(u, v, ());
            }
            return Some(sub);
        } else {
            // 独立なら基底に追加
            b = trial;
            basis_edges.push(eidx);
        }
    }
    None
}

/// =============== クロージャ計算（Python: compute_closure_from_an_instance） ===============

pub fn compute_closure_from_an_instance(
    complete_g: &UnGraphMap<usize, ()>,
    h: &UnGraphMap<usize, ()>,
    t: usize,
    seed: Option<u64>,
) -> UnGraphMap<usize, ()> {
    let nodes: Vec<usize> = complete_g.nodes().sorted().collect();
    let n = nodes.len();
    let index: HashMap<usize, usize> =
        nodes.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();

    let p = random_normal_matrix(n, t, seed);
    let d = t * (t + 1) / 2;

    let inst_edges = stable_edges(h);
    let mut mtx = zeros_matrix(d, inst_edges.len());
    for (col, &(u, v)) in inst_edges.iter().enumerate() {
        let iu = *index.get(&u).unwrap();
        let iv = *index.get(&v).unwrap();
        let mut sym = outer(&p[iu], &p[iv]);
        let out_vu = outer(&p[iv], &p[iu]);
        add_inplace(&mut sym, &out_vu);
        let col_vec = flatten_upper_tri(&sym);
        for r in 0..d {
            mtx[r][col] = col_vec[r];
        }
    }

    let inst_set: HashSet<(usize, usize)> = inst_edges.iter().cloned().collect();
    let outside_edges: Vec<(usize, usize)> = stable_edges(complete_g)
        .into_iter()
        .filter(|e| !inst_set.contains(e))
        .collect();

    let mut closure_edges = inst_edges.clone();
    let mut m_cur = mtx;

    for (u, v) in outside_edges {
        let iu = *index.get(&u).unwrap();
        let iv = *index.get(&v).unwrap();
        let mut sym = outer(&p[iu], &p[iv]);
        let out_vu = outer(&p[iv], &p[iu]);
        add_inplace(&mut sym, &out_vu);
        let col_vec = flatten_upper_tri(&sym);

        let trial = concat_columns(&m_cur, &col_vec);
        let old_rank = matrix_rank(&m_cur);
        let new_rank = matrix_rank(&trial);
        if old_rank == new_rank {
            // スパン内にある => クロージャに追加
            m_cur = trial;
            closure_edges.push((u, v));
        }
    }

    let mut out = UnGraphMap::<usize, ()>::new();
    for &u in &nodes {
        out.add_node(u);
    }
    for (u, v) in closure_edges {
        out.add_edge(u, v, ());
    }
    out
}

/// =============== 二部グラフ判定と二部集合の取得 ===============

fn bipartition_sets(g: &UnGraphMap<usize, ()>) -> Option<(HashSet<usize>, HashSet<usize>)> {
    let mut color: HashMap<usize, bool> = HashMap::new(); // false / true の2色
    for start in g.nodes() {
        if color.contains_key(&start) {
            continue;
        }
        // BFS
        let mut stack = vec![start];
        color.insert(start, false);
        while let Some(u) = stack.pop() {
            let cu = *color.get(&u).unwrap();
            for v in g.neighbors(u) {
                if !color.contains_key(&v) {
                    color.insert(v, !cu);
                    stack.push(v);
                } else if color[&v] == cu {
                    return None; // 同色隣接 => 非二部
                }
            }
        }
    }
    let mut a = HashSet::new();
    let mut b = HashSet::new();
    for (v, c) in color {
        if c {
            b.insert(v);
        } else {
            a.insert(v);
        }
    }
    Some((a, b))
}

/// =============== C_{n,t} 判定（Python: is_in_C_n_t） ===============

pub fn is_in_c_n_t(mut closure: UnGraphMap<usize, ()>, t: usize) -> bool {
    // 孤立ノードを削除
    loop {
        let isolates: Vec<usize> = closure
            .nodes()
            .filter(|&v| closure.neighbors(v).next().is_none())
            .collect();
        if isolates.is_empty() {
            break;
        }
        for v in isolates {
            closure.remove_node(v);
        }
    }

    let n = closure.node_count();

    // t=1 の端ケース
    if t == 1 {
        let e = closure.edge_count();
        return e == n * (n.saturating_sub(1)) / 2 || e == 0;
    }

    // 次数 n-1 のノードがあれば取り除いて t-1 へ再帰
    if let Some((v, _)) = closure
        .nodes()
        .map(|v| (v, closure.neighbors(v).count()))
        .find(|&(_v, deg)| deg == n.saturating_sub(1))
    {
        closure.remove_node(v);
        return is_in_c_n_t(closure, t - 1);
    }

    // 辺が無ければ True
    if closure.edge_count() == 0 {
        return true;
    }

    // 二部グラフ可否と |U|,|V| による分岐
    if let Some((u_set, v_set)) = bipartition_sets(&closure) {
        let (u, v) = (u_set.len(), v_set.len());
        let e = closure.edge_count();

        match t {
            6 => {
                if (u == 3 && v == 5) || (u == 5 && v == 3) {
                    return e == 15;
                } else if (u == 4 && v == 5) || (u == 5 && v == 4) {
                    return e == 20;
                } else if u == 4 && v == 4 {
                    return e == 16;
                } else if u == 5 && v == 5 {
                    return e == 25;
                }
            }
            5 => {
                if (u == 3 && v == 4) || (u == 4 && v == 3) {
                    return e == 12;
                } else if u == 4 && v == 4 {
                    return e == 16;
                }
            }
            4 => {
                if u == 3 && v == 3 {
                    return e == 9;
                }
            }
            _ => {}
        }
        // 条件に該当しなければ偽（デバッグ出力は呼び出し側で）
        false
    } else {
        // 非二部のケース：Python 版同様、ここでは偽を返す
        false
    }
}

/// =============== t=6 用の総合チェック（Python: check_statement_t_6） ===============

// === 追加: ログ収集版 ===
pub fn check_statement_t_6(graphs: &[UnGraphMap<usize, ()>], comment: &str) -> Vec<String> {
    let mut logs: Vec<String> = Vec::new();

    let pb = ProgressBar::new(graphs.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "Checking the statement for t = 6: {msg} [{bar:40.cyan/blue}] {pos}/{len}",
        )
        .unwrap(),
    );
    pb.set_message(comment.to_string());

    for (idx, g_orig) in graphs.iter().enumerate() {
        let original_nodes: Vec<usize> = g_orig.nodes().sorted().collect();

        let mut g = g_orig.clone();
        loop {
            let isolates: Vec<usize> = g
                .nodes()
                .filter(|&v| g.neighbors(v).next().is_none())
                .collect();
            if isolates.is_empty() {
                break;
            }
            for v in isolates {
                g.remove_node(v);
            }
        }

        if g.nodes().any(|v| g.neighbors(v).count() > 5) {
            pb.inc(1);
            continue;
        }
        if g.edge_count() > 21 {
            pb.inc(1);
            continue;
        }

        let rank = symmetric_tensor_matroid_rank_graph(&g, 6, Some(42));
        if rank == g.edge_count() {
            pb.inc(1);
            continue;
        }

        let circuit = match find_one_circuit_in_symmetric_tensor_matroid(&g, 6, Some(42)) {
            Some(h) => h,
            None => {
                logs.push(format!(
                    "This may be a counter example (no circuit found but rank deficient)\nG's rank: {}\n|V|={}, |E|={}\n",
                    rank, g.node_count(), g.edge_count()
                ));
                pb.inc(1);
                continue;
            }
        };

        let mut complete = UnGraphMap::<usize, ()>::new();
        for &u in &original_nodes {
            complete.add_node(u);
        }
        for i in 0..original_nodes.len() {
            for j in (i + 1)..original_nodes.len() {
                complete.add_edge(original_nodes[i], original_nodes[j], ());
            }
        }

        let closure = compute_closure_from_an_instance(&complete, &circuit, 6, Some(42));

        if !is_in_c_n_t(closure.clone(), 6) {
            logs.push(format!(
                "This may be a counter example\nG's rank: {}\nG: |V|={}, |E|={}\nclosure: |V|={}, |E|={}\n",
                rank,
                g.node_count(),
                g.edge_count(),
                closure.node_count(),
                closure.edge_count()
            ));
        }

        pb.inc(1);
        if (idx + 1) % 100 == 0 {
            pb.tick();
        }
    }
    pb.finish();

    logs
}
