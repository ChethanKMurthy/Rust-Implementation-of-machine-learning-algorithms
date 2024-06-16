use ndarray::{Array1, Array2};
use std::collections::BinaryHeap;

pub struct KNearestNeighbors {
    k: usize,
    train_x: Array2<f64>,
    train_y: Array1<f64>,
}

impl KNearestNeighbors {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            train_x: Array2::zeros((0, 0)),
            train_y: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) {
        self.train_x = x;
        self.train_y = y;
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.outer_iter().map(|sample| self.predict_sample(&sample)).collect()
    }

    fn predict_sample(&self, sample: &Array1<f64>) -> f64 {
        let mut heap = BinaryHeap::new();
        for (i, train_sample) in self.train_x.outer_iter().enumerate() {
            let distance = (&train_sample - sample).mapv(|v| v * v).sum::<f64>().sqrt();
            heap.push((distance, self.train_y[i]));
            if heap.len() > self.k {
                heap.pop();
            }
        }
        heap.iter().map(|&(_, label)| label).sum::<f64>() / self.k as f64
    }
}

fn main() {
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
    let y = x.slice(s![.., 0]).mapv(|v| if v > 50.0 { 1.0 } else { 0.0 });

    let mut model = KNearestNeighbors::new(3);
    model.fit(x, y);

    let test_x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
    let predictions = model.predict(&test_x);
    println!("Predictions: {:?}", predictions);
}
