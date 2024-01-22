use crate::activation_functions::Sigmoid;
use crate::loss_functions::RootMeanSquared;
use crate::neuron::{Layer, Net, Neuron};

pub fn make_net_from_online_example() -> Net<Sigmoid, Sigmoid, RootMeanSquared> {
    let l0_n0 = Neuron::from_weights(vec![0.15, 0.2], 0.35);
    let l0_n1 = Neuron::from_weights(vec![0.25, 0.3], 0.35);
    let lo_n0 = Neuron::from_weights(vec![0.4, 0.45], 0.6);
    let lo_n1 = Neuron::from_weights(vec![0.5, 0.55], 0.6);
    let l0 = Layer::from_neurons(vec![l0_n0, l0_n1]);
    let lo = Layer::from_neurons(vec![lo_n0, lo_n1]);

    let mut net = Net::<Sigmoid, Sigmoid, RootMeanSquared>::from_layers(vec![l0], lo);
    net
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use crate::neuron::{Label, LayerType, NeuronTrait, WeightType};
    use super::*;


    #[test]
    fn test_first_forward() {
        let mut net = make_net_from_online_example();
        let inputs = vec![0.05, 0.1];
        let first_pass_expected = vec![0.75136507, 0.7729285];
        let actual = net.forward_pass(&inputs);
        assert_eq!(actual, &first_pass_expected);
        let training_1 = vec![0.01, 0.99];
        // Currently we have definitely RMS without the half, the example uses the half, however
        // we take the average and the example doesn't so we get the 2.0 back again
        let expected_error = 0.298371109;
        let actual_error = net.evaluate_loss(&inputs, &training_1, false);
        assert_eq!(actual_error, expected_error);
    }

    #[test]
    fn test_backprop() {
        let mut net = make_net_from_online_example();
        let inputs = vec![0.05, 0.1];
        let training_1 = vec![0.01, 0.99];
        // Currently we have definitely RMS without the half, the example uses the half, however
        // we take the average and the example doesn't so we get the 2.0 back again
        let expected_error = 0.298371109;
        let actual_error = net.evaluate_loss(&inputs, &training_1, true);
        net.back_propagate(&training_1);
        let expected_gradient5 = 0.082167041;
        let actual_gradient5 = net.output_layer.neurons[0].weight_gradients[0];
        assert_eq!(actual_gradient5, expected_gradient5);
        // Example uses learning rate of 0.5
        let alpha = 0.5;
        let expected_new_output_weights = vec![0.35891648f32, 0.408666186, 0.511301270, 0.561370121];
        let expected_output_gradients = expected_new_output_weights.iter().enumerate().map(|(i, w)| {
            let (div, modu) = (i / 2, i % 2);
            (net.output_layer.neurons[div].weights[modu] - w) / alpha
        }).collect::<Vec<f32>>();
        // The example only uses one weight per layer, so we remove our bias weights and bias gradients
        let labels = net.labels();
        let output_labels = labels.into_iter().filter(|l| (l.layer == LayerType::Output) && (l.weight != WeightType::Bias)).collect::<Vec<Label>>();
        let output_indices = output_labels.iter().map(|label| net.index_for_label(label)).collect::<Vec<usize>>();
        let grads = net.gradient_vector().clone();
        let weights = net.weight_vector().clone();
        let calced_grads = output_indices.iter().map(|i| grads[*i]).collect::<Vec<f32>>();
        let calced_weights = output_indices.iter().map(|i| weights[*i]).collect::<Vec<f32>>();

        // TODO: Fix this and maybe make a crate
        for i in 0..calced_grads.len() {
            assert_approx_eq!(calced_grads[i], expected_output_gradients[i], 0.0001);
        }

        let actual_new_output_weights = calced_grads.iter().zip(calced_weights.iter()).map(|(g, w)| {
            w - alpha * g
        }).collect::<Vec<f32>>();
        for i in 0..calced_weights.len() {
            assert_approx_eq!(calced_weights[i], expected_new_output_weights[i], 0.0001);
        }


        let expected_new_hidden_weights = vec![0.149780716, 0.19956143, 0.24975114, 0.29950229];
    }
}