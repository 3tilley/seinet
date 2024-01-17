

pub trait LossFunction {
    fn loss(expected: &Vec<f32>, actual: &Vec<f32>) -> f32;
    fn derivative(expected: &Vec<f32>, actual: &Vec<f32>) -> Vec<f32>;
}
pub struct RootMeanSquared;

impl LossFunction for RootMeanSquared {
    fn loss(expected: &Vec<f32>, actual: &Vec<f32>) -> f32 {
        expected.iter().zip(actual).map(|(ex, act)| {
            (act - ex).powi(2)
        }).sum::<f32>() / (expected.len() as f32)
    }

    fn derivative(expected: &Vec<f32>, actual: &Vec<f32>) -> Vec<f32> {
        let n = expected.len() as f32;
        expected.iter().zip(actual).map(|(ex, act)| {
            2.0 * (act - ex) / n
        }).collect()
    }
}