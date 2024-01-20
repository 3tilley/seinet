use std::thread::sleep;
use std::time::Duration;
use plotly::common::{Line, Mode, Title};
use plotly::layout::{Axis, LayoutGrid};
use plotly::{Layout, Plot, Scatter, Trace};
use tracing_subscriber::fmt::format;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use crate::activation_functions::{ActivationFunction, Relu};
use crate::fitting::{BasicHarness, Progress};
use crate::loss_functions::{LossFunction, RootMeanSquared};
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

fn make_trace_from_weights<T: ActivationFunction, U: ActivationFunction, V: LossFunction>(inputs: &Vec<f32>, progress: &Progress) -> Vec<Box<Line>>{
    Line::f
    let outputs = inputs.iter().map(|inp| net.forward_pass(&vec![*inp]).clone()).collect::<Vec<Vec<f32>>>();
    let mut scatters = Vec::new();
    for i in 0..1 {
        let scat = Scatter::new(inputs.clone(), outputs.iter().map(|output| output[0]).collect()).name(format!("y{}_net", i));
        scatters.push(scat);
    }
    scatters
}

fn stacked_subplots(traces: Vec<Box<Scatter<f32, f32>>>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new().grid(
        LayoutGrid::new()
            .columns(1)
            .rows(traces.len())
    );
    plot.add_traces(traces.into_iter().map(|t| t as Box<dyn Trace>).collect());
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
    let outputs = inputs.iter().map(|inp| (vec![*inp], vec![relu_2_3b(*inp)])).collect::<Vec<_>>();

    let mut net = Net::<Relu, Relu, RootMeanSquared>::new(&mut rng, 1, 1, vec![]);
    let mut basic = BasicHarness::new(net, outputs, 0.01);
    basic.train_n_or_converge(1000);

    // Plotly
    let trace = Scatter::new(basic.progress.weights.iter().map(|w| w[0]).collect(), basic.progress.errors.clone()).mode(Mode::Markers).line(Line::new().color("blue"));

    let layout = Layout::new().x_axis(Axis::new().title(Title::from("X Axis")))
        .y_axis(Axis::new().title(Title::from("Y Axis")))
        .title(Title::from("My Plot"));

    // Weights
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout.clone());
    // plot.show();

    let mut out_net = basic.net;

    // Real func vs net
    let mut traces = make_trace_from_output(&inputs, &mut out_net);
    let actual = Scatter::new(inputs.clone(), inputs.iter().map(|inp| relu_2_3b(*inp)).collect()).name(format!("y{}_actual", 1));
    traces.push(actual);
    let plot_2 = stacked_subplots(traces);
    plot_2.show();
    sleep(Duration::from_secs(5));
}
