use ndarray::{Array1, Array2};
use rand::distr::{Distribution, Uniform};

#[derive(Debug)]
pub struct NeuralNet {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl NeuralNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let uniform = Uniform::new(-0.5f32, 0.5f32).unwrap();

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
        let a2 = softmax(z2);

        (z1, a1, a2)
    }

    pub fn backward(
        &self,
        x: &Array1<f32>,
        y_true: &Array1<f32>,
        mut z1: Array1<f32>,
        a1: Array1<f32>,
        mut a2: Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
        // dL/dz2 = a2 - y_true
        a2 -= y_true;
        let dz2 = a2;

        // dL/dW2 = dz2 * a1.T
        let dw2 = dz2.clone().insert_axis(ndarray::Axis(1)) * a1.insert_axis(ndarray::Axis(0));

        // dL/da1 = W2.T * dz2
        let da1 = self.w2.t().dot(&dz2);

        // dL/db2 = dz2
        let db2 = dz2;

        // dL/dz1 = da1 * ReLU'(z1)
        z1.zip_mut_with(&da1, |z, &da| *z = da * ((*z > 0.0) as u8 as f32));
        let dz1 = z1;

        let dw1 = dz1.clone().insert_axis(ndarray::Axis(1)) * x.clone().insert_axis(ndarray::Axis(0));
        // dL/db1 = dz1
        let db1 = dz1;

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
        self.w1.scaled_add(-lr, dw1);
        self.b1.scaled_add(-lr, db1);
        self.w2.scaled_add(-lr, dw2);
        self.b2.scaled_add(-lr, db2);
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

fn softmax(mut z: Array1<f32>) -> Array1<f32> {
    let max_z = z.fold(f32::NEG_INFINITY, |acc, &z| acc.max(z));
    let mut sum_exp_z = 0.0f32;
    z.map_inplace(|z| { let v = (*z - max_z).exp(); sum_exp_z += v; *z = v; });
    z.map_inplace(|x| *x /= sum_exp_z);
    z
}
