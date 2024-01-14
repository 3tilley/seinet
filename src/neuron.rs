use crate::activation_functions::ActivationFunction;

pub struct Neuron<T: ActivationFunction> {
    input_weights: Vec<f32>,
    activation_function: T,
    bias: f32,
}

impl<T: ActivationFunction> Neuron<T> {
    pub fn evaluate_neuron<T: ActivationFunction>(&self, input: &Vec<f32>) -> f32 {
        assert_eq!(input.len(), self.input_weights.len());
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += self.activation_function.activate(input[i] * self.input_weights[i] + self.bias)
        }
        sum
    }
}

pub struct Layer<T> {
    neurons: Vec<Neuron<T>>,
    output: Vec<f32>,
}

impl<T> Layer<T> {
    pub fn evaluate_layer(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        for (i, neuron) in self.neurons.iter().enumerate() {
            self.output[i] = neuron.evaluate_neuron(input);
        }
        &self.output
    }
}

pub struct Net {
    layers: Vec<Layer>,
    loss_function: fn(&Vec<f32>) -> f32,
}

impl Net {
    pub fn evaluate(&mut self, input: Vec<f32>) -> &Vec<f32> {
        let mut working_vec = &input;
        for mut layer in self.layers {
            working_vec = layer.evaluate_layer(working_vec)
        }
        working_vec
    }

    pub fn evaluate_loss(&mut self, input: Vec<f32>) -> f32 {
        let predicted = self.evaluate(input);
        (self.loss_function)(predicted)
    }
}
