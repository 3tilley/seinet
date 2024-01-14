pub trait Input: Into<Vec<f32>> {


}

pub trait Output: From<usize> {
    fn size() -> usize;
}

