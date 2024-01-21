
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub trait ActivationFunction: Default {
    fn activate(x: f32) -> f32;
    fn derivative(x: f32) -> f32;
}

#[derive(Default)]
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

#[derive(Default)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f32) -> f32 {
        let sig = Sigmoid::activate(x);
        sig * (1.0 - sig)
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(Relu::activate(0.0), 0.0);
        assert_eq!(Relu::activate(1.0), 1.0);
        assert_eq!(Relu::activate(-1.0), 0.0);
    }

    #[test]
    fn test_relu_deriv() {
        assert_eq!(Relu::derivative(0.0), 0.0);
        assert_eq!(Relu::derivative(1.0), 1.0);
        assert_eq!(Relu::derivative(-1.0), 0.0);
    }
}
