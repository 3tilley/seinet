use std::cmp::min;
use rand::prelude::SliceRandom;
use tracing::{debug, info};
use crate::activation_functions::{ActivationFunction, Relu};
use crate::loss_functions::{LossFunction, RootMeanSquared};
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


#[derive(Copy, Clone, Debug)]
pub enum TerminationCondition {
    Epochs(usize, f32),
    Loss(usize, f32),
}

#[derive(Copy, Clone, Debug)]
pub struct TerminationCriteria {
    pub epochs: u64,
    pub loss_limit: f32,
}

impl TerminationCriteria {
    pub fn new(epochs: u64, loss_limit: f32) -> TerminationCriteria {
        TerminationCriteria {
            epochs,
            loss_limit,
        }
    }

    pub fn is_complete(&self, epochs: usize, loss: f32) -> Option<TerminationCondition> {
        if epochs >= self.epochs as usize {
            Some(TerminationCondition::Epochs(epochs, loss))
        } else if loss < self.loss_limit {
            Some(TerminationCondition::Loss(epochs, loss))
        } else {
            None
        }
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
        // debug!("Weights: {:?}", self.weights);
        // debug!("Gradients: {:?}", self.gradients);
    }

}

pub struct BatchParameters {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last_if_smaller: bool,
}

pub struct BasicHarness<T: ActivationFunction, V: ActivationFunction, W: LossFunction> {
    pub net: Net<T, V, W>,
    pub training_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub testing_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub running_averages: RunningAverage,
    pub progress: Progress,
    loss_limit: f32,
    learning_rate: f32,
    termination: TerminationCriteria,
    batch_params: BatchParameters,
    rng: rand::rngs::ThreadRng,
}
impl<T: ActivationFunction, V: ActivationFunction, W: LossFunction> BasicHarness<T, V, W> {
    pub fn new(net: Net<T, V, W>, mut input_data: Vec<(Vec<f32>, Vec<f32>)>, train_frac: f32, learning_rate: f32, termination: TerminationCriteria, batch_params: BatchParameters) -> BasicHarness<T, V, W> {
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
            learning_rate,
            progress: Progress::new(),
            termination,
            batch_params,
            rng
        }
    }

    pub fn evaluate_and_store(&mut self, input: &Vec<f32>, output: &Vec<f32>) -> f32 {
        let loss = self.net.evaluate_loss(input, output, true);
        self.running_averages.update(self.net.gradient_vector());
        loss
    }

    pub fn train_epoch(&mut self) -> (bool, f32) {
        // An epoch is all training data for one cycle
        if self.batch_params.shuffle {
            self.training_data.shuffle(&mut self.rng);
        }
        let mut num_batches = self.training_data.len() / self.batch_params.batch_size;
        if (num_batches % self.batch_params.batch_size != 0) && !self.batch_params.drop_last_if_smaller {
            num_batches += 1;
        }
        let mut loss = 0.0;
        for i in 0..num_batches {
            let start = i * self.batch_params.batch_size;
            let end = min(start + self.batch_params.batch_size, self.training_data.len());
            // TODO: Avoid this clone
            let batch = &self.training_data[start..end].to_vec();
            for (input, output) in batch {
                loss += self.evaluate_and_store(input, output);
            }
            loss /= self.batch_params.batch_size as f32;
            self.update_weights();
            self.running_averages.reset();
        }
        info!("Loss: {}", loss);
        let converged = loss < self.loss_limit;
        self.progress.update(loss, &self.net.weight_vector().clone(), &self.net.gradient_vector());
        if !converged {
            self.update_weights();
            self.running_averages.reset();
        }
        (converged, loss)
    }

    pub fn train_n_or_converge(&mut self) {
        let mut epochs = 0;
        let mut finished = None;
        while finished.is_none() {
            epochs += 1;
            info!("Epoch: {}", epochs);
            let (converged, loss) = self.train_epoch();
            finished = self.termination.is_complete(epochs, loss);
        }

        match finished {
            Some(TerminationCondition::Epochs(epochs, loss)) => info!("Epoch limit reached: {} epochs. Loss: {}", epochs, loss),
            Some(TerminationCondition::Loss(epochs, loss)) => info!("Converged after {} epochs. loss {}", epochs, loss),
            None => unreachable!("Shouldn't exit loop unless converged"),
        }
    }

    pub fn update_weights(&mut self) {
        // TOOD: Think of the resetting the averages and avoiding this clone
        let weight_deltas = self.running_averages.values.iter().map(|v| -v * self.learning_rate).collect::<Vec<_>>();
        self.net.update_weights(&weight_deltas);
    }
}