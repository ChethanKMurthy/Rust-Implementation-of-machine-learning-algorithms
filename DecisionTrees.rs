use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TreeNode {
    feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: Option<f64>,
}

impl TreeNode {
    fn new() -> Self {
        TreeNode {
            feature: None,
            threshold: None,
            left: None,
            right: None,
            value: None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.value.is_some()
    }
}

pub struct DecisionTree {
    root: TreeNode,
    max_depth: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        DecisionTree {
            root: TreeNode::new(),
            max_depth,
        }
    }

    fn gini(&self, y: &Array1<f64>) -> f64 {
        let mut counts = HashMap::new();
        for &value in y.iter() {
            *counts.entry(value).or_insert(0.0) += 1.0;
        }
        counts.values().map(|&count| {
            let p = count / y.len() as f64;
            p * (1.0 - p)
        }).sum()
    }

    fn best_split(&self, x: &Array2<f64>, y: &Array1<f64>) -> (usize, f64) {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gini = std::f64::MAX;

        for feature in 0..x.ncols() {
            let mut thresholds: Vec<f64> = x.slice(s![.., feature]).to_vec();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for &threshold in thresholds.iter() {
                let left_indices: Vec<usize> = x.outer_iter()
                    .enumerate()
                    .filter(|(_, row)| row[feature] <= threshold)
                    .map(|(i, _)| i)
                    .collect();
                let right_indices: Vec<usize> = x.outer_iter()
                    .enumerate()
                    .filter(|(_, row)| row[feature] > threshold)
                    .map(|(i, _)| i)
                    .collect();

                let left_y = y.select(Axis(0), &left_indices);
                let right_y = y.select(Axis(0), &right_indices);

                let left_gini = self.gini(&left_y);
                let right_gini = self.gini(&right_y);

                let gini = (left_gini * left_y.len() as f64 + right_gini * right_y.len() as f64) / y.len() as f64;

                if gini < best_gini {
                    best_gini = gini;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }
        (best_feature, best_threshold)
    }

    fn build_tree(&self, x: &Array2<f64>, y: &Array1<f64>, depth: usize) -> TreeNode {
        if depth >= self.max_depth || y.len() <= 1 || self.gini(y) == 0.0 {
            let mut node = TreeNode::new();
            let most_common_value = y.iter().fold(HashMap::new(), |mut acc, &value| {
                *acc.entry(value).or_insert(0) += 1;
                acc
            }).into_iter().max_by_key(|&(_, count)| count).unwrap().0;
            node.value = Some(most_common_value);
            return node;
        }

        let (best_feature, best_threshold) = self.best_split(x, y);

        let left_indices: Vec<usize> = x.outer_iter()
            .enumerate()
            .filter(|(_, row)| row[best_feature] <= best_threshold)
            .map(|(i, _)| i)
            .collect();
        let right_indices: Vec<usize> = x.outer_iter()
            .enumerate()
            .filter(|(_, row)| row[best_feature] > best_threshold)
            .map(|(i, _)| i)
            .collect();

        let left_x = x.select(Axis(0), &left_indices);
        let left_y = y.select(Axis(0), &left_indices);
        let right_x = x.select(Axis(0), &right_indices);
        let right_y = y.select(Axis(0), &right_indices);

        let mut node = TreeNode::new();
        node.feature = Some(best_feature);
        node.threshold = Some(best_threshold);
        node.left = Some(Box::new(self.build_tree(&left_x, &left_y, depth + 1)));
        node.right = Some(Box::new(self.build_tree(&right_x, &right_y, depth + 1)));

        node
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.root = self.build_tree(x, y, 0);
    }

    fn predict_sample(&self, node: &TreeNode, sample: &Array1<f64>) -> f64 {
        if node.is_leaf() {
            return node.value.unwrap();
        }
        let feature = node.feature.unwrap();
        let threshold = node.threshold.unwrap();
        if sample[feature] <= threshold {
            self.predict_sample(node.left.as_ref().unwrap(), sample)
        } else {
            self.predict_sample(node.right.as_ref().unwrap(), sample)
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.outer_iter().map(|sample| self.predict_sample(&self.root, &sample)).collect()
    }
}

fn main() {
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
    let y = x.slice(s![.., 0]).mapv(|v| if v > 50.0 { 1.0 } else { 0.0 });

    let mut model = DecisionTree::new(3);
    model.fit(&x, &y);

    let predictions = model.predict(&x);
    println!("Predictions: {:?}", predictions);
}
