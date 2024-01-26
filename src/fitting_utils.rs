use tracing::debug;

pub struct RunningAverage {
    pub num_averages: usize,
    pub size: usize,
    pub values: Vec<f32>,
}

impl RunningAverage {
    pub fn new(size: usize) -> RunningAverage {
        RunningAverage {
            num_averages: 0,
            size,
            values: vec![0.0; size],
        }
    }

    pub fn update(&mut self, values: &Vec<f32>) {
        assert_eq!(values.len(), self.size);
        self.num_averages += 1;
        self.values.iter_mut().zip(values).for_each(|(running, new)| {
            *running += (new - *running) / (self.num_averages as f32);
        })
    }

    pub fn reset(&mut self) {
        self.num_averages = 0;
        self.values.fill(0.0);
    }
}


#[derive(Copy, Clone, Debug)]
pub enum TerminationCondition {
    Epochs(usize, f32),
    Loss(usize, f32),
}

#[derive(Copy, Clone, Debug)]
pub struct TerminationCriteria {
    pub epochs: u64,
    pub loss_limit: f32,
}

impl TerminationCriteria {
    pub fn new(epochs: u64, loss_limit: f32) -> TerminationCriteria {
        TerminationCriteria {
            epochs,
            loss_limit,
        }
    }

    pub fn is_complete(&self, epochs: usize, loss: f32) -> Option<TerminationCondition> {
        if epochs >= self.epochs as usize {
            Some(TerminationCondition::Epochs(epochs, loss))
        } else if loss < self.loss_limit {
            Some(TerminationCondition::Loss(epochs, loss))
        } else {
            None
        }
    }
}

pub struct Progress {
    pub errors: Vec<f32>,
    // Vector of timesteps, each of which is a vector of weights
    pub weights: Vec<Vec<f32>>,
    pub gradients: Vec<Vec<f32>>,
}

impl Progress {
    pub fn new() -> Progress {
        Progress {
            errors: Vec::new(),
            weights: vec![],
            gradients: vec![],
        }
    }

    pub fn update(&mut self, errors: f32, weights: &Vec<f32>, gradients: &Vec<f32>) {
        self.errors.push(errors);
        self.weights.push(weights.clone());
        self.gradients.push(gradients.clone());
        debug!("Weights: {:?}", self.weights);
        debug!("Gradients: {:?}", self.gradients);
    }

}

pub struct BatchParameters {
    pub batch_size: Option<usize>,
    pub shuffle: bool,
    pub drop_last_if_smaller: bool,
}

impl BatchParameters {
    pub fn new(batch_size: usize, shuffle: bool, drop_last_if_smaller: bool) -> BatchParameters {
        BatchParameters {
            batch_size: Some(batch_size),
            shuffle,
            drop_last_if_smaller,
        }
    }
    pub fn one_batch(shuffle: bool) -> BatchParameters {
        BatchParameters {
            batch_size: None,
            shuffle,
            drop_last_if_smaller: false,
        }
    }

    pub fn iterator(&self, training_len: usize) -> BatchParametersIterator {
        let batch_size = self.batch_size.unwrap_or(training_len);
        BatchParametersIterator::new(batch_size, training_len, self.drop_last_if_smaller)
    }
}


pub struct BatchParametersIterator {
    batch_size: usize,
    num_regular_batches: usize,
    current_index: usize,
    last_batch_size: Option<(usize)>,
}

impl BatchParametersIterator {
    pub fn new(batch_size: usize, training_len: usize, drop_last_if_smaller: bool) -> BatchParametersIterator {
        assert!(training_len >= batch_size);
        let num_regular_batches = training_len / batch_size;
        BatchParametersIterator {
            batch_size,
            num_regular_batches,
            current_index: 0,
            last_batch_size: (!drop_last_if_smaller).then_some(training_len % batch_size).filter(|&size| size > 0),
        }
    }
}

impl Iterator for BatchParametersIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // 10: [4, 4, 2] -> [(0..4), (4..8), (8..10)]
        if self.current_index < self.num_regular_batches {
            let start = self.current_index * self.batch_size;
            let end = start + self.batch_size;
            self.current_index += 1;
            Some((start, end))
        } else if self.current_index == self.num_regular_batches {
            self.last_batch_size.map(|size| {
                let start = self.current_index * self.batch_size;
                let end = start + size;
                self.current_index += 1;
                (start, end)
            })
        } else {
            None
        }
    }
}

pub trait DataManager {
    fn training_data(&self) -> Vec<(Vec<f32>, Vec<f32>)>;
    fn validation_data(&self) -> Vec<(Vec<f32>, Vec<f32>)>;
    fn testing_data(&self) -> Vec<(Vec<f32>, Vec<f32>)>;


}

pub struct SimpleDataManager {
    pub training_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub validation_data: Vec<(Vec<f32>, Vec<f32>)>,
    pub testing_data: Vec<(Vec<f32>, Vec<f32>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_iterator() {
        let batch_size = 4;
        let training_len = 10;
        let drop_last_if_smaller = false;
        let mut iter = BatchParametersIterator::new(batch_size, training_len, drop_last_if_smaller);
        assert_eq!(iter.next(), Some((0, 4)));
        assert_eq!(iter.next(), Some((4, 8)));
        assert_eq!(iter.next(), Some((8, 10)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_batch_iterator_with_vectors() {
        let batch_size = 4;
        let training_len = 10;
        let drop_last_if_smaller = false;
        let mut iter = BatchParametersIterator::new(batch_size, training_len, drop_last_if_smaller);
        let vec = vec![1,2,3,4,5,6,7,8,9,10];
        let mut group_vec = Vec::new();
        while let Some((start, end)) = iter.next() {
            group_vec.push(vec[start..end].to_vec());
        }
        assert_eq!(group_vec, vec![vec![1,2,3,4], vec![5,6,7,8], vec![9,10]]);
    }

    #[test]
    fn test_batch_iterator_smaller() {
        let batch_size = 4;
        let training_len = 10;
        let drop_last_if_smaller = true;
        let mut iter = BatchParametersIterator::new(batch_size, training_len, drop_last_if_smaller);
        assert_eq!(iter.next(), Some((0, 4)));
        assert_eq!(iter.next(), Some((4, 8)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_batch_iterator_none() {
        let batch_size = 4;
        let training_len = 10;
        let drop_last_if_smaller = true;
        let batch_params = BatchParameters::one_batch(true);
        let mut iter = batch_params.iterator(10);
        assert_eq!(iter.next(), Some((0, 10)));
        assert_eq!(iter.next(), None);
    }
}