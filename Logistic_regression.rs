use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::Rng;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct LogisticRegression {
    weights: Array1<f64>,
    bias: f64,
}

impl LogisticRegression {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::random_using(input_size, Uniform::new(-1.0, 1.0), &mut rng);
        let bias = rng.gen_range(-1.0..1.0);
        Self { weights, bias }
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            let linear_model = x.dot(&self.weights) + self.bias;
            let predictions = Self::sigmoid(&linear_model);
            let errors = y - &predictions;
            let gradient = x.t().dot(&errors) / x.nrows() as f64;
            self.weights += &(gradient * learning_rate);
            self.bias += errors.mean().unwrap() * learning_rate;
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let linear_model = x.dot(&self.weights) + self.bias;
        Self::sigmoid(&linear_model)
    }
}

fn main() {
    let x = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
    let y = x.slice(s![.., 0]).mapv(|v| if v > 50.0 { 1.0 } else { 0.0 });

    let mut model = LogisticRegression::new(2);
    model.train(&x, &y, 1000, 0.01);

    let predictions = model.predict(&x);
    println!("Predictions: {:?}", predictions);
}
