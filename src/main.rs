use std::thread::sleep;
use std::time::Duration;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use crate::activation_functions::Relu;
use crate::fitting::BasicHarness;
use crate::loss_functions::RootMeanSquared;
use crate::neuron::Net;

mod activation_functions;
mod neuron;
// mod dqn;
mod vectors;
mod fitting;
mod loss_functions;
mod float_utils;

fn relu_2_3b(x: f32) -> f32 {
    let weighted_sum = 2.0 * x + 3.0;
    if weighted_sum > 0.0 {
        weighted_sum
    } else {
        0.0
    }
}

fn main() {
    println!("Starting training...");
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
    let mut inputs = Vec::new();
    let start = -5.0;
    let end = 5.0;
    let step = 0.1;
    let mut current = start;
    while current <= end {
        inputs.push(current);
        current += step;
    }
    let mut rng = rand::thread_rng();
    let outputs = inputs.iter().map(|inp| (vec![*inp], vec![relu_2_3b(*inp)])).collect::<Vec<_>>();
    let mut net = Net::<Relu, Relu, RootMeanSquared>::new(&mut rng, 1, 1, vec![]);
    // let net = Net<Relu, Relu, RootMeanSquared>::new()
    let mut basic = BasicHarness::new(net, outputs, 0.01);
    basic.train_n_or_converge(100000);
}
