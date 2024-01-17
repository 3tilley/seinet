use std::ops::Neg;
use rand::Rng;
use crate::neuron::Net;
use crate::vectors::{Input, Output};

pub struct QLearningHyperParameters {
    pub learning_rate: f32,
    pub gamma: f32,
    pub exploration_probability: f32,
    pub exploration_decay: f32,
    pub batch_size: u64,
    pub memory_buffer: u64,
    pub episodes: u64,
    pub max_iterations_per_episode: u64,
}

pub struct Episode<T: Input, V: Output> {
    pub current_state: T,
    pub action: V,
    pub reward: f32,
    pub next_state: T,
    pub done: bool,
}

pub struct DeepQLearning<T, V, W, X, Y> {
    pub net: Net<W, X, Y>,
    pub state_size: usize,
    pub action_size: usize,
    pub exploration_probability: f32,
    rng: rand::rngs::ThreadRng,
    pub hyper_params: QLearningHyperParameters,
    pub experiences: ringbuffer::AllocRingBuffer<Episode<T, V>>,
}

impl<T: Input, V: Output, W, X, Y> DeepQLearning<T, V, W, X, Y> {
    pub fn compute_action(&mut self, current_state: T) {
        let explore = self.rng.gen_bool(self.exploration_probability as f64);
        if explore {
            V::from(self.rng.gen_range(0..V::size()))
        } else {
            let output = self.net.evaluate(current_state.into());
            let ind =output.iter().enumerate().max_by_key(|v| v.1).unwrap();
            V::from(ind.0)
        }
    }

    pub fn update_exploration_probability(&mut self) {
        self.exploration_probability *= self.hyper_params.exploration_decay.neg().exp();
    }

}

