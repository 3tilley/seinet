

pub trait LossFunction {
    fn loss(expected: &Vec<f32>, actual: &Vec<f32>) -> f32;
    fn derivative(x: &Vec<f32>);
}
pub struct RootMeanSquared;

impl LossFunction for RootMeanSquared {
    fn loss(expected: &Vec<f32>, actual: &Vec<f32>) -> f32 {
        expected.iter().zip(actual).map(|(ex, act)| {
            (act - ex).powi(2)
        }).sum()
    }

    fn derivative(x: &Vec<f32>) {
        todo!()
    }
}