use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct LinearRegression {
    weights: Array1<f64>,
    bias: f64,
}

impl LinearRegression {
    pub fn new(input_size: usize) -> Self {
        let weights = Array1::random(input_size, Uniform::new(-1.0, 1.0));
        let bias = 0.0; // Initialize bias to 0
        Self { weights, bias }
    }

    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            let predictions = x.dot(&self.weights) + self.bias;
            let errors = &predictions - y;
            let gradient = x.t().dot(&errors) / x.nrows() as f64;
            self.weights -= &gradient * learning_rate;
            self.bias -= errors.mean().unwrap() * learning_rate;
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }
}

fn main() {
    // Sample data
    let sizes = vec![100.0, 150.0, 200.0, 250.0, 300.0]; // House sizes in square meters
    let prices = vec![150000.0, 200000.0, 250000.0, 300000.0, 350000.0]; // House prices in dollars

    let x = Array2::from_shape_vec((sizes.len(), 1), sizes).unwrap();
    let y = Array1::from_shape_vec((prices.len(),), prices).unwrap();

    // Initialize and train the model
    let mut model = LinearRegression::new(1);
    model.train(&x, &y, 1000, 0.0001); // Adjust epochs and learning rate as needed

    // Make predictions
    let sizes_to_predict = vec![120.0, 180.0, 220.0]; // Sizes of houses to predict prices
    let x_predict = Array2::from_shape_vec((sizes_to_predict.len(), 1), sizes_to_predict).unwrap();
    let predictions = model.predict(&x_predict);
    
    // Display predictions
    println!("House Price Predictions:");
    for (i, prediction) in predictions.iter().enumerate() {
        println!("House size: {} sqm, Predicted Price: ${:.2}", sizes_to_predict[i], prediction);
    }
}
