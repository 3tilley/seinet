
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub trait ActivationFunction {
    fn activate(x: f32) -> f32;
    fn derivative(x: f32) -> f32;
}

pub struct Relu;

impl ActivationFunction for Relu {
    fn activate(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}