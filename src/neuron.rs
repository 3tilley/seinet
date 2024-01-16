use std::ffi::IntoStringError;
use crate::activation_functions::ActivationFunction;

pub struct Neuron<T: ActivationFunction> {
    weights: Vec<f32>,
    gradients: Vec<f32>,
    activation_function: T,
    bias: f32,
    last_preactivation: f32,
}

impl<T: ActivationFunction> Neuron<T> {
    pub fn evaluate_neuron<T: ActivationFunction>(&mut self, input: &Vec<f32>) -> f32 {
        assert_eq!(input.len(), self.weights.len());
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += input[i] * self.weights[i] + self.bias;
        }
        self.last_preactivation = sum;
        self.activation_function.activate(sum)
    }

    pub fn update_gradients<T: ActivationFunction>(&mut self, next_layer_delta: &Vec<f32>) {
        let mut sum = 0;

        self.activation_function.derivative(self.last_preactivation);
    }
}

pub struct Layer<T> {
    neurons: Vec<Neuron<T>>,
    output: Vec<f32>,
}

impl<T> Layer<T> {
    pub fn evaluate_layer(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        for (i, mut neuron) in self.neurons.iter().enumerate() {
            self.output[i] = neuron.evaluate_neuron(input);
        }
        &self.output
    }
}

// pub trait Layer {
//    fn evaluate_layer(&mut self, input: &Vec<f32>) -> &Vec<f32>;
// }
//
// impl<T> Layer for GenericLayer<T> {
//     fn evaluate_layer(&mut self, input: &Vec<f32>) -> &Vec<f32> {
//         self.evaluate_layer(input)
//     }
// }

pub struct Net<T, V> {
    pub layers: Vec<Layer<T>>,
    pub output_layer: Layer<V>,
    pub loss_function: fn(&Vec<f32>) -> f32,
    pub back_propped: bool,
}

impl<T, V> Net<T, V> {
    pub fn forward_pass(&mut self, input: Vec<f32>) -> &Vec<f32> {
        self.back_propped = false;
        let mut working_vec = &input;
        for mut layer in self.layers {
            working_vec = layer.evaluate_layer(working_vec)
        }
        self.output_layer.evaluate_layer(working_vec)
    }

    pub fn back_propagate(&mut self) {
        self.back_propped = true;


    }

    pub fn evaluate_loss(&mut self, input: Vec<f32>) -> f32 {
        let predicted = self.evaluate(input);
        (self.loss_function)(predicted)
    }
}
