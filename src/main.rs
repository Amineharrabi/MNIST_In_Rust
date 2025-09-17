mod model;
mod utils;

use csv::ReaderBuilder;
use model::NeuralNet;
use ndarray::Array1;
use std::error::Error;
use utils::{accuracy, display_image, one_hot_encode};

fn load_mnist_csv(path: &str) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result?;
        let label: usize = record.get(0).unwrap().parse()?;
        let pixels: Vec<f32> = record
            .iter()
            .skip(1)
            .map(|s| s.parse::<f32>().unwrap() / 255.0)
            .collect();

        let x = Array1::from(pixels);
        let y = one_hot_encode(label, 10);

        xs.push(x);
        ys.push(y);
    }

    Ok((xs, ys))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (train_xs, train_ys) = load_mnist_csv("data/mnist_train.csv")?;
    let (test_xs, test_ys) = load_mnist_csv("data/mnist_test.csv")?;

    let mut net = NeuralNet::new(784, 64, 10);

    let epochs = 1;
    let lr = 0.01;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (i, (x, y_true)) in train_xs.iter().zip(train_ys.iter()).enumerate() {
            let (z1, a1, a2) = net.forward(x);

            let loss = -y_true
                .iter()
                .zip(a2.iter())
                .map(|(&t, &p)| t * p.ln())
                .sum::<f32>();
            total_loss += loss;

            let pred = a2
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            let true_class = y_true.iter().enumerate().find(|(_, &v)| v > 0.5).unwrap().0;
            if pred == true_class {
                correct += 1;
            }

            let (dw1, db1, dw2, db2) = net.backward(x, y_true, &z1, &a1, &a2);

            net.update(&dw1, &db1, &dw2, &db2, lr);

            if i % 1000 == 0 {
                println!(
                    "Sample {}: Loss = {:.4}, Acc = {:.2}%",
                    i,
                    loss,
                    (correct as f32 / (i + 1) as f32) * 100.0
                );
            }
        }

        let avg_loss = total_loss / train_xs.len() as f32;
        let train_acc = correct as f32 / train_xs.len() as f32;
        println!(
            "Epoch {}: Avg Loss = {:.4}, Train Acc = {:.2}%",
            epoch,
            avg_loss,
            train_acc * 100.0
        );
    }

    let mut test_preds = Vec::new();
    let mut test_labels = Vec::new();

    let demo_index = 42;
    let mut demo_x: Option<Array1<f32>> = None;
    let mut demo_y_true: Option<usize> = None;

    for (i, (x, y_true)) in test_xs.iter().zip(test_ys.iter()).enumerate() {
        let (_, _, a2) = net.forward(x);
        let pred = a2
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let true_class = y_true.iter().enumerate().find(|(_, &v)| v > 0.5).unwrap().0;

        test_preds.push(pred);
        test_labels.push(true_class);

        // Capture the demo sample
        if i == demo_index {
            demo_x = Some(x.clone());
            demo_y_true = Some(true_class);
        }
    }

    let test_acc = accuracy(&test_preds, &test_labels);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);

    if let (Some(x), Some(true_label)) = (demo_x, demo_y_true) {
        println!("\n=== SINGLE SAMPLE PREDICTION DEMO ===");
        display_image(&x);

        let (predicted, probabilities) = net.predict_single(&x);

        println!("True Label: {}", true_label);
        println!("Predicted: {}", predicted);
        println!("Confidence per class:");
        for (i, &prob) in probabilities.iter().enumerate() {
            println!("  {}: {:.2}%", i, prob * 100.0);
        }

        if predicted == true_label {
            println!("\n CORRECT ");
        } else {
            println!("\n WRONG");
        }
    }

    Ok(())
}
