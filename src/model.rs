use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};

pub struct NeuralNet {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl NeuralNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-0.5f32, 0.5f32);

        let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| uniform.sample(&mut rng));
        let b1 = Array1::zeros(hidden_size);
        let w2 = Array2::from_shape_fn((output_size, hidden_size), |_| uniform.sample(&mut rng));
        let b2 = Array1::zeros(output_size);

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // z1 = W1 * x + b1
        let z1 = self.w1.dot(x) + &self.b1;
        // a1 = ReLU(z1)
        let a1 = z1.mapv(|v| v.max(0.0));
        // z2 = W2 * a1 + b2
        let z2 = self.w2.dot(&a1) + &self.b2;
        // a2 = softmax(z2)
        let a2 = softmax(&z2);

        (z1, a1, a2)
    }

    pub fn backward(
        &self,
        x: &Array1<f32>,
        y_true: &Array1<f32>,
        z1: &Array1<f32>,
        a1: &Array1<f32>,
        a2: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
        let _output_size = a2.len();
        let _hidden_size = a1.len();

        // dL/dz2 = a2 - y_true
        let dz2 = &(a2 - y_true);

        // dL/dW2 = dz2 * a1.T
        let dw2 = dz2.clone().insert_axis(ndarray::Axis(1)) * a1.clone().insert_axis(ndarray::Axis(0));
        // dL/db2 = dz2
        let db2 = dz2.clone();

        // dL/da1 = W2.T * dz2
        let da1 = self.w2.t().dot(dz2);

        // dL/dz1 = da1 * ReLU'(z1)
        let dz1 = &da1 * z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        let dw1 = dz1.clone().insert_axis(ndarray::Axis(1)) * x.clone().insert_axis(ndarray::Axis(0));
        // dL/db1 = dz1
        let db1 = dz1.clone();

        (dw1, db1, dw2, db2)
    }

    pub fn update(
        &mut self,
        dw1: &Array2<f32>,
        db1: &Array1<f32>,
        dw2: &Array2<f32>,
        db2: &Array1<f32>,
        lr: f32,
    ) {
        self.w1 = &self.w1 - &(lr * dw1);
        self.b1 = &self.b1 - &(lr * db1);
        self.w2 = &self.w2 - &(lr * dw2);
        self.b2 = &self.b2 - &(lr * db2);
    }

    pub fn predict_single(&self, x: &Array1<f32>) -> (usize, Array1<f32>) {
    let (_, _, a2) = self.forward(x);
    let predicted_class = a2
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    (predicted_class, a2)
}
}

fn softmax(z: &Array1<f32>) -> Array1<f32> {
    let max_z = z.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_z = z.mapv(|x| (x - max_z).exp());
    let sum_exp_z = exp_z.sum();
    exp_z / sum_exp_z
}
