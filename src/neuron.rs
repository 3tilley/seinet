
pub struct Neuron {
    input_weights: Vec<f32>,
    activation_function: fn(f32) -> f32,
}

impl Neuron {
    pub fn pass(&self, input: Vec<f32>) -> f32 {
        assert_eq!(input.len(), self.input_weights.len());
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += (self.activation_function)(input[i] * self.input_weights[i])
        }
        sum
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {

}

pub struct Net {
    layers: Vec<Layer>,
}
