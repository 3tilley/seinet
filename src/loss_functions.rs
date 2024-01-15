
pub struct RootMeanSquared;

pub trait LossFunction {
    fn loss(x: &Vec<f32>);
    fn derivative(x: &Vec<f32>, )
}