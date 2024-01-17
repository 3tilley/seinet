use rand::Rng;
use crate::activation_functions::ActivationFunction;
use crate::loss_functions::LossFunction;

pub struct Neuron<T: ActivationFunction> {
    activation_function: T,
    weights: Vec<f32>,
    bias: f32,
    weight_gradients: Vec<f32>,
    bias_gradient: f32,
    last_preactivation: f32,
    last_output: f32,
}

impl<T: ActivationFunction> Neuron<T> {

    pub fn new(rng: &mut impl Rng, input_len: usize) -> Neuron<T> {
        let mut weights = Vec::with_capacity(input_len);
        for _ in 0..input_len {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        Neuron {
            activation_function: T::default(),
            weights,
            bias: rng.gen_range(-1.0..1.0),
            weight_gradients: vec![0.0; input_len],
            bias_gradient: 0.0,
            last_preactivation: 0.0,
            last_output: 0.0,
        }
    }

    pub fn from_weights(weights: Vec<f32>, bias: f32) -> Neuron<T> {
        Neuron {
            weight_gradients: vec![0.0; weights.len()],
            activation_function: T::default(),
            weights,
            bias,
            bias_gradient: 0.0,
            last_preactivation: 0.0,
            last_output: 0.0,
        }
    }
    pub fn evaluate_neuron(&mut self, input: &Vec<f32>) -> f32 {
        assert_eq!(input.len(), self.weights.len());
        let mut sum = self.bias;
        for i in 0..input.len() {
            sum += input[i] * self.weights[i];
        }
        self.last_preactivation = sum;
        let output = T::activate(sum);
        self.last_output = output;
        output
    }

    pub fn update_gradients(&mut self, next_layer_delta: &Vec<f32>) {
        let mut sum = 0;

        // self.activation_function.derivative(self.last_preactivation);
    }
}

pub struct Layer<T: ActivationFunction> {
    neurons: Vec<Neuron<T>>,
    output: Vec<f32>,
}

impl<T: ActivationFunction> Layer<T> {

    pub fn new(size: usize, input_len: usize, rng: &mut impl rand::Rng) -> Layer<T> {
        let mut neurons = Vec::with_capacity(size);
        for _ in 0..size {
            neurons.push(Neuron::new(rng, input_len));
        }
        Layer { neurons, output: vec![0.0; size] }
    }

    pub fn from_neurons(neurons: Vec<Neuron<T>>) -> Layer<T> {
        Layer { output: vec![0.0; neurons.len()], neurons  }
    }

    pub fn evaluate_layer(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        for (i, mut neuron) in self.neurons.iter_mut().enumerate() {
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

pub struct Net<T: ActivationFunction, V: ActivationFunction, W: LossFunction> {
    pub layers: Vec<Layer<T>>,
    pub output_layer: Layer<V>,
    pub loss_function: W,
    pub back_propped: bool,
    pub last_input: Vec<f32>,
}

impl<T: ActivationFunction, V: ActivationFunction, W: LossFunction> Net<T, V, W> {
    pub fn forward_pass(&mut self, input: Vec<f32>) -> &Vec<f32> {
        self.back_propped = false;
        self.last_input = input;
        let mut working_vec = &self.last_input;
        for mut layer in self.layers.iter_mut() {
            working_vec = layer.evaluate_layer(working_vec)
        }
        self.output_layer.evaluate_layer(working_vec)
    }

    pub fn back_propagate(&mut self, expected: &Vec<f32>) {
        self.back_propped = true;
        // This represents: delta = L'(y, y*) * H'(z)
        // where L is the loss function and H is the activation function
        // Initially it's the errors for the last layer but it will be reused
        let mut errors = {
            let mut intermediate = W::derivative(expected, &self.output_layer.output);
            for i in 0..intermediate.len() {
                intermediate[i] *= V::derivative(self.output_layer.neurons[i].last_preactivation);
            }
            intermediate
        };
        let previous_activation = match self.layers.last() {
            None => &self.last_input,
            Some(layer) => &layer.output,
        };
        for (j, mut neuron) in self.output_layer.neurons.iter_mut().enumerate() {
            neuron.bias_gradient = errors[j];
            for j in 0..neuron.weight_gradients.len() {
                neuron.weight_gradients[j] = errors[j] * previous_activation[j];
            }
        }
        let mut layer_ahead = &self.output_layer;
        for (i, mut layer) in self.layers.iter_mut().enumerate().rev() {
            // i runs from (layers.len() - 1) to 0
            let previous_activation = if i == 0 {
                &self.last_input
            } else {
                &layer.output
            };
            let activation_prime = layer.neurons.iter().map(|n| T::derivative(n.last_preactivation)).collect::<Vec<_>>();
            errors = activation_prime.into_iter().enumerate().map(|(j, h_prime)| {
                let mut weight_sum = 0.0;
                for (k,ahead_neuron) in layer_ahead.neurons.iter().enumerate() {
                    weight_sum += ahead_neuron.weights[j] * errors[k];
                }
                weight_sum * h_prime
            }).collect();
            for (j, mut neuron) in layer.neurons.iter_mut().enumerate() {
                neuron.bias_gradient = errors[j];
                for j in 0..neuron.weights.len() {
                    neuron.weight_gradients[j] = errors[j] * previous_activation[j];
                }
            }
        }

    }

    pub fn evaluate_loss(&mut self, input: Vec<f32>, expected: Vec<f32>) -> f32 {
        let predicted = self.forward_pass(input);
        W::loss(&expected, predicted)
    }
}

#[cfg(test)]
mod tests {
    use crate::activation_functions::Relu;
    use super::*;

    #[test]
    fn test_neuron() {
        let w = vec![2.0,];
        let b = 3.0;
        let mut neuron = Neuron::<Relu>::from_weights(w, b);
        let inputs = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
        let outputs = inputs.iter().map(|inp| neuron.evaluate_neuron(&vec![*inp,])).collect::<Vec<_>>();
        let expected_outputs = vec![0.0, 0.0, 0.0, 0.0, 1.0, 3.0];
        assert_eq!(outputs, expected_outputs);
    }

    #[test]
    fn test_layer() {
        let w = vec![2.0,];
        let b = 3.0;
        let mut neuron = Neuron::<Relu>::from_weights(w, b);
        let mut layer = Layer::from_neurons(vec![neuron]);
        let inputs = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
        let outputs = inputs.iter().map(|inp| layer.evaluate_layer(&vec![*inp,]).clone()).collect::<Vec<_>>();
        let expected_outputs = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 3.0].into_iter().map(|o| vec![o]).collect::<Vec<_>>();
        assert_eq!(outputs, expected_outputs);
    }

    #[test]
    fn test_net() {
        let w = vec![2.0,];
        let b = 3.0;
        let mut neuron = Neuron::<Relu>::from_weights(w, b);
        let mut layer = Layer::from_neurons(vec![neuron]);
        let inputs = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
        let outputs = inputs.iter().map(|inp| layer.evaluate_layer(&vec![*inp,]).clone()).collect::<Vec<_>>();
        let expected_outputs = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 3.0].into_iter().map(|o| vec![o]).collect::<Vec<_>>();
        assert_eq!(outputs, expected_outputs);
    }

}