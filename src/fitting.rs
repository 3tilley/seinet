use rand::prelude::SliceRandom;
use tracing::{debug, info};
use crate::activation_functions::Relu;
use crate::loss_functions::RootMeanSquared;
use crate::neuron::Net;

struct MiniBatchSgdHyperParameters {
    pub learning_rate: f32,
    epochs: u64,
    batch_size: u64,
}
struct SgdHyperParameters {
    pub learning_rate: f32,
}

struct StochasticGradientDescent {

}

struct GradientDescent {

}

pub trait FittingStrategy {
    
}

pub struct RunningAverage {
    pub num_averages: usize,
    pub size: usize,
    pub values: Vec<f32>,
}

impl RunningAverage {
    pub fn new(size: usize) -> RunningAverage {
        RunningAverage {
            num_averages: 0,
            size,
            values: vec![0.0; size],
        }
    }

    pub fn update(&mut self, values: &Vec<f32>) {
        assert_eq!(values.len(), self.size);
        self.num_averages += 1;
        self.values.iter_mut().zip(values).for_each(|(running, new)| {
            *running += (new - *running) / (self.num_averages as f32);
        })
    }

    pub fn reset(&mut self) {
        self.num_averages = 0;
        self.values.fill(0.0);
    }
}

pub struct Progress {
    pub errors: Vec<f32>,
    pub weights: Vec<Vec<f32>>,
    pub gradients: Vec<Vec<f32>>,
}

impl Progress {
    pub fn new() -> Progress {
        Progress {
            errors: Vec::new(),
            weights: vec![],
            gradients: vec![],
        }
    }

    pub fn update(&mut self, errors: f32, weights: &Vec<f32>, gradients: &Vec<f32>) {
        self.errors.push(errors);
        self.weights.push(weights.clone());
        self.gradients.push(gradients.clone());
        debug!("Weights: {:?}", self.weights);
        debug!("Gradients: {:?}", self.gradients);
    }

}

pub struct BasicHarness {
    pub net: Net<Relu, Relu, RootMeanSquared>,
    pub training_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub testing_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub running_averages: RunningAverage,
    pub progress: Progress,
    loss_limit: f32,
    learning_rate: f32,
}

impl BasicHarness {
    pub fn new(net: Net<Relu, Relu, RootMeanSquared>, mut input_data: Vec<(Vec<f32>, Vec<f32>)>, train_frac: f32) -> BasicHarness {
        let mut rng = rand::thread_rng();
        input_data.shuffle(&mut rng);
        let end_train_index = ((input_data.len() as f32) * train_frac) as usize;
        let (training, testing) = input_data.split_at(end_train_index);
        BasicHarness {
            running_averages: RunningAverage::new(net.total_weights),
            net,
            training_data: training.into(),
            testing_data: testing.into(),
            loss_limit: 0.00001,
            learning_rate: 0.1,
            progress: Progress::new(),
        }
    }

    pub fn evaluate_and_store(&mut self, input: &Vec<f32>, output: &Vec<f32>) -> f32 {
        let loss = self.net.evaluate_loss(input, output, true);
        self.running_averages.update(self.net.gradient_vector());
        loss
    }

    pub fn train_epoch(&mut self) -> bool {
        // An epoch is all training data for one cycle
        let mut loss = 0.0;
        for i in 0..self.training_data.len() {
            // TODO: Fix this to prevent cloning
            let (input, output) = &self.training_data[i].clone();
            loss += self.evaluate_and_store(input, output);
        }
        loss /= self.training_data.len() as f32;
        info!("Loss: {}", loss);
        let total_loss = loss < self.loss_limit;
        self.progress.update(loss, &self.net.weight_vector().clone(), &self.net.gradient_vector());
        self.update_weights();
        self.running_averages.reset();
        total_loss
    }

    pub fn train_n_or_converge(&mut self, n: usize) {
        let mut epochs = 0;
        let mut converged = false;
        while !converged && epochs < n {
            epochs += 1;
            info!("Epoch: {}", epochs);
            converged = self.train_epoch();
        }
    }

    pub fn update_weights(&mut self) {
        // TOOD: Think of the resetting the averages and avoiding this clone
        let gradient_deltas = self.running_averages.values.iter().map(|v| -v * self.learning_rate).collect::<Vec<_>>();
        self.net.update_weights(&gradient_deltas);
    }
}