use std::thread::sleep;
use std::time::Duration;
use plotly::common::{Line, Mode, Title};
use plotly::layout::{Axis, LayoutGrid};
use plotly::{Configuration, Layout, Plot, Scatter, Trace};
use plotly::layout::GridPattern::Independent;
use tracing::info;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use crate::activation_functions::{ActivationFunction, LeakyRelu, Relu, Sigmoid};
use crate::fitting::{BasicHarness, BatchParameters, Progress, TerminationCriteria};
use crate::loss_functions::{LossFunction, RootMeanSquared};
use crate::neuron::{Label, Net};

mod activation_functions;
mod neuron;
// mod dqn;
mod vectors;
mod fitting;
mod loss_functions;
mod float_utils;
mod online_example;

fn relu_2_3b(x: f32) -> f32 {
    let weighted_sum = 2.0 * x + 3.0;
    if weighted_sum > 0.0 {
        weighted_sum
    } else {
        0.0
    }
}

// This isn't quite a sawtooth, it's meant to look something like the below
// ___/\____
fn sawtoothish(x: f32) -> f32 {
    if x < -2.0 {
        -1.0
    } else if x < 0.0 {
        2.0 * x + 3.0
    } else if x < 2.0 {
        -2.0 * x + 3.0
    } else {
        -1.0
    }
}

fn make_trace(inputs: &Vec<f32>, func: impl Fn(f32) -> f32) -> Box<Scatter<f32, f32>> {
    let outputs = inputs.iter().map(|inp| func(*inp)).collect();
    Scatter::new(inputs.clone(), outputs).mode(Mode::LinesMarkers).line(Line::new().color("blue"))
}

fn make_trace_vec(inputs: &Vec<f32>, mut func: impl FnMut(f32) -> &'static Vec<f32>) -> Vec<Box<Scatter<f32, f32>>> {
    let outputs = inputs.iter().map(|inp| func(*inp).clone()).collect::<Vec<Vec<f32>>>();
    let scatters = outputs.iter().map(|out_vec| Scatter::new(inputs.clone(), out_vec.clone()));
    scatters.collect()
}

fn make_trace_from_output<T: ActivationFunction, U: ActivationFunction, V: LossFunction>(inputs: &Vec<f32>, net: &mut Net<T, U, V>) -> Vec<Box<Scatter<f32, f32>>>{
    let outputs = inputs.iter().map(|inp| net.forward_pass(&vec![*inp]).clone()).collect::<Vec<Vec<f32>>>();
    let mut scatters = Vec::new();
    for i in 0..1 {
        let scat = Scatter::new(inputs.clone(), outputs.iter().map(|output| output[0]).collect()).name(format!("y{}_net", i));
        scatters.push(scat);
    }
    scatters
}

fn make_trace_from_weights(inputs: &Vec<f32>, labels: Vec<Label>, progress: &Progress) -> Vec<Vec<Box<Scatter<f32, f32>>>>{
    let xs = (0..progress.weights.len()).map(|i| i as f32).collect::<Vec<f32>>();
    let mut scatters = Vec::new();
    scatters.push(vec![Scatter::new(xs.clone(), progress.errors.clone()).name("error").mode(Mode::LinesMarkers).line(Line::new().color("red"))]);
    let mut trans_weights = vec![Vec::new(); progress.weights[0].len()];
    let mut trans_grads = vec![Vec::new(); progress.weights[0].len()];
    for (_, (timestep)) in progress.weights.iter().zip(&progress.gradients).enumerate() {
        // Timesteps is a vector of timesteps, each of which is a vector of weights and a vector of gradients
        for (j,(weight)) in timestep.0.iter().enumerate() {
            trans_weights[j].push(*weight);
            trans_grads[j].push(timestep.1[j]);
        }
    }
    scatters.push(vec![]);
    scatters.push(vec![]);
    for (w, label) in trans_weights.iter().zip(&labels) {
        let scat = Scatter::new(xs.clone(),w.clone()).name(label.to_string());
        scatters[1].push(scat);
    }

    for (w, label) in trans_grads.iter().zip(labels) {
        let scat = Scatter::new(xs.clone(),w.clone()).name(format!("{}_grad",label.to_string()));
        scatters[2].push(scat);
    }
    assert_eq!(scatters.len(), 3);
    scatters
}

fn stacked_subplots(traces: Vec<Vec<Box<Scatter<f32, f32>>>>) -> Plot {
    let mut plot = Plot::new();
    let config = Configuration::new().autosizable(true).fill_frame(true);
    plot.set_configuration(config);
    let layout = Layout::new().grid(
        LayoutGrid::new()
            .columns(1)
            .rows(traces.len())
            .pattern(Independent)
            .y_gap(0.5)
    );
    for (i, group) in traces.into_iter().enumerate() {
        let x_axis = format!("x{}", i+1);
        let y_axis = format!("y{}", i+1);
        plot.add_traces(group.into_iter().map(|t| {
            t.x_axis(&x_axis).y_axis(&y_axis) as Box<dyn Trace>
        }).collect());
    }
    plot.set_layout(layout);
    plot
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
    let outputs = inputs.iter().map(|inp| (vec![*inp], vec![sawtoothish(*inp)])).collect::<Vec<_>>();

    let mut net = Net::<Relu, Sigmoid, RootMeanSquared>::new(&mut rng, 1, 1, vec![4,4]);
    let term = TerminationCriteria::new(10000, 0.00001);
    let batch_params = BatchParameters { batch_size: 8, shuffle: true, drop_last_if_smaller: true};
    let mut basic = BasicHarness::new(net, outputs.clone(), 0.8, 0.2, term, batch_params);
    basic.train_n_or_converge();


    // Real func vs net
    let plot = make_training_plots(&mut inputs, outputs, &mut basic);

    info!("Weights: {:?}" , basic.progress.weights.last().unwrap());
    info!("Gradients: {:?}" , basic.progress.gradients.last().unwrap());
    sleep(Duration::from_secs(1));
}

fn make_training_plots<T: ActivationFunction, V: ActivationFunction, W: LossFunction>(inputs: &mut Vec<f32>, outputs: Vec<(Vec<f32>, Vec<f32>)>, harness: &mut BasicHarness<T, V, W>) {
    let net_labels = harness.net.labels();


    let mut traces = make_trace_from_output(&inputs, &mut harness.net);
    let (actual_inputs, actual_outputs) = outputs.clone().into_iter().map(|(input, output)| (input[0], output[0])).unzip();
    let actual = Scatter::new(actual_inputs, actual_outputs).name(format!("y{}_actual", 1));
    let (training_inputs, training_outputs) = harness.training_data.clone().into_iter().map(|(input, output)| (input[0], output[0])).unzip();
    let training = Scatter::new(training_inputs, training_outputs).name("training").mode(Mode::Markers).line(Line::new().color("green"));
    traces.push(training);
    traces.push(actual);
    let mut total_plots = vec![traces];
    let weights_trace = make_trace_from_weights(&inputs, net_labels, &harness.progress);
    total_plots.extend(weights_trace);
    let plot_2 = stacked_subplots(total_plots);
    plot_2.show();
}
