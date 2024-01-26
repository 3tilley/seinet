use rand::prelude::{SliceRandom, StdRng};
use tracing::{debug, info};
use crate::activation_functions::ActivationFunction;
use crate::fitting_utils::{BatchParameters, Progress, RunningAverage, TerminationCondition, TerminationCriteria};
use crate::loss_functions::LossFunction;
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


pub struct TrainingHarnessBuilder<T: ActivationFunction, V: ActivationFunction, W: LossFunction> {
    net: Net<T, V, W>,
    training_data: Vec<(Vec<f32>, Vec<f32>)>,
    train_frac: f32,
    termination: TerminationCriteria,
    batch_params: BatchParameters,
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
    rng: rand::rngs::StdRng,
}
impl<T: ActivationFunction, V: ActivationFunction, W: LossFunction> BasicHarness<T, V, W> {
    pub fn new(net: Net<T, V, W>, mut input_data: Vec<(Vec<f32>, Vec<f32>)>, train_frac: f32, learning_rate: f32, termination: TerminationCriteria, batch_params: BatchParameters, mut rng: StdRng) -> BasicHarness<T, V, W> {
        input_data.shuffle(&mut rng);
        let end_train_index = ((input_data.len() as f32) * train_frac) as usize;
        let (training, testing) = input_data.split_at(end_train_index);
        assert!(training.len() > 0);
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
        let mut loss = 0.0;
        for (start, end) in self.batch_params.iterator(self.training_data.len()) {
            // debug!("Batch: {}", i);
            // TODO: Avoid this clone
            let batch = &self.training_data[start..end].to_vec();
            self.running_averages.reset();
            for (input, output) in batch {
                // let num_grads = self.net.numerical_gradients(input, output);
                loss += self.evaluate_and_store(input, output);
                let grads = self.net.gradient_vector().clone();
                // let labels = self.net.labels();
                // for (i, ((grad, num_grad), label)) in grads.iter().zip(num_grads.iter()).zip(labels).enumerate() {
                //     // assert_approx_eq!(grad, num_grad, 0.001);
                //     // println!("Grad_{}: {} {}", label.to_string(), grad, num_grad);
                // }
                // println!("Back-propped: {:?}", self.net.gradient_vector());
                // println!("Numerical:    {:?}", self.net.numerical_gradients(input, output));
                // self.net.evaluate_loss(input, output, true);
            }
            loss /= self.batch_params.batch_size.unwrap_or(self.training_data.len()) as f32;
            self.update_weights();
        }
        // info!("Loss: {}", loss);
        let converged = loss < self.loss_limit;
        self.progress.update(loss, &self.net.weight_vector().clone(), &self.net.gradient_vector());
        // if !converged {
        //     self.update_weights();
        //     self.running_averages.reset();
        // }
        (converged, loss)
    }

    pub fn train_n_or_converge(&mut self) {
        let mut epochs = 0;
        let mut finished = None;
        while finished.is_none() {
            epochs += 1;
            // info!("Epoch: {}", epochs);
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
        // TODO: Think of the resetting the averages and avoiding this clone
        let weight_deltas = self.running_averages.values.iter().map(|v| -v * self.learning_rate).collect::<Vec<_>>();
        debug!("Updating weights by: {:?}", weight_deltas);
        self.net.update_weights(&weight_deltas);
    }
}