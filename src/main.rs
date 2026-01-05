mod model;
mod utils;

use csv::ReaderBuilder;
use model::NeuralNet;
use ndarray::Array1;
use std::error::Error;
use std::time::Instant;
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

        xs.push(Array1::from(pixels));
        ys.push(one_hot_encode(label, 10));
    }

    Ok((xs, ys))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (train_xs, train_ys) = load_mnist_csv("data/mnist_train.csv")?;
    let (test_xs, test_ys) = load_mnist_csv("data/mnist_test.csv")?;
    let xlen = train_xs.first().unwrap().len(); //784 = 28^2
    let ylen = train_ys.first().unwrap().len(); //10
    let mut net = NeuralNet::new(xlen, 4, ylen);

    let start_tme = Instant::now();

    let epochs = 5;
    const LR: f32 = 0.01;
    const CHUNK_SIZE: usize = 10_000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        train_xs.chunks(CHUNK_SIZE).zip(train_ys.chunks(CHUNK_SIZE))
            .enumerate().for_each(|(chunk_idx, (xs_chunk, ys_chunk))| {

            let mut loss = 0.0;
            xs_chunk.iter().zip(ys_chunk.iter()).for_each(|(x, y_true)| {
                let (z1, a1, a2) = net.forward(x);
                loss = -y_true
                    .iter()
                    .zip(a2.iter())
                    .map(|(&t, &p)| t * p.ln())
                    .sum::<f32>();
                total_loss += loss;

                let pred = a2
                    .indexed_iter()
                    .max_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                let true_class = y_true.iter().position(|&v| v > 0.5).unwrap();
                correct += (pred == true_class) as usize;

                let (dw1, db1, dw2, db2) = net.backward(&x, &y_true, z1, a1, a2);

                net.update(&dw1, &db1, &dw2, &db2, LR);
            });

            println!(
                "Sample {}: Loss = {:.4}, Acc = {:.2}%",
                (chunk_idx + 1) * CHUNK_SIZE,
                loss,
                (correct as f32 / ((chunk_idx + 1) * CHUNK_SIZE + 1) as f32) * 100.0
                );
        });

        let avg_loss = total_loss / train_xs.len() as f32;
        let train_acc = correct as f32 / train_xs.len() as f32;
        println!(
            "Epoch {}: Avg Loss = {:.4}, Train Acc = {:.2}%",
            epoch,
            avg_loss,
            train_acc * 100.0
        );
    }

    let mut test_preds = Vec::with_capacity(test_xs.len());
    let mut test_labels = Vec::with_capacity(test_xs.len());

    let demo_index = 42;
    let mut demo: Option<(Array1<f32>, usize)> = None;

    test_xs.iter().zip(test_ys.iter()).enumerate().for_each(|(i, (x, y_true))| {
        let (_, _, a2) = net.forward(x);
        let pred = a2
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let true_class = y_true.iter().position(|&v| v > 0.5).unwrap();

        test_preds.push(pred);
        test_labels.push(true_class);

        // Capture the demo sample
        if i == demo_index {
            demo = Some((x.clone(), true_class));
        }
    });

    println!("Total duration: {:.2?}ms", start_tme.elapsed().as_secs_f64()*1e3);

    let test_acc = accuracy(&test_preds, &test_labels);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);

    if let Some((x, true_label)) = demo {
        println!("\n=== SINGLE SAMPLE PREDICTION DEMO ===");
        display_image(&x);

        let (predicted, probabilities) = net.predict_single(&x);

        println!("True Label: {}", true_label);
        println!("Predicted: {}", predicted);
        println!("Confidence per class:");
        probabilities.iter().enumerate().for_each(|(i, &prob)| {
            println!("  {}: {:>5.2}%", i, prob * 100.0);
        });

        println!("\n {}", if predicted == true_label { "CORRECT" } else { "WRONG" });
    }

    Ok(())
}
