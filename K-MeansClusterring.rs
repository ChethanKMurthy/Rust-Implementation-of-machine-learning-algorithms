use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct KMeans {
    centroids: Array2<f64>,
}

impl KMeans {
    pub fn new(n_clusters: usize, n_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let centroids = Array2::random_using((n_clusters, n_features), Uniform::new(0.0, 1.0), &mut rng);
        Self { centroids }
    }

    fn closest_centroids(&self, x: &Array2<f64>) -> Array1<usize> {
        x.outer_iter()
            .map(|row| {
                self.centroids.outer_iter()
                    .enumerate()
                    .map(|(i, centroid)| (i, (&row - &centroid).mapv(|v| v * v).sum()))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn compute_centroids(&self, x: &Array2<f64>, labels: &Array1<usize>, n_clusters: usize) -> Array2<f64> {
        let mut centroids = Array2::zeros((n_clusters, x.ncols()));
        for i in 0..n_clusters {
            let cluster_points = x.select(Axis(0), &labels.iter().enumerate()
                .filter(|(_, &label)| label == i)
                .map(|(i, _)| i)
                .collect::<Vec<_>>());
            centroids.slice_mut(s![i, ..]).assign(&cluster_points.mean_axis(Axis(0)).unwrap());
        }
        centroids
    }

    pub fn fit(&mut self, x: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            let labels = self.closest_centroids(x);
            self.centroids = self.compute_centroids(x, &labels, self.centroids.nrows());
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        self.closest_centroids(x)
    }
}

fn main() {
    let x = Array2::random((100, 2), Uniform::new(0.0, 10.0));

    let mut model = KMeans::new(3, 2);
    model.fit(&x, 100);

    let predictions = model.predict(&x);
    println!("Predictions: {:?}", predictions);
}
