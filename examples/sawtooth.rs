
// This isn't quite a sawtooth, it's meant to look something like the below
// ___/\____
fn sawtoothish(x: f32) -> f32 {
    if x < -2.0 {
        -1.0
    } else if x < 0.0 {
        2.0 * x + 4.0
    } else if x < 2.0 {
        -2.0 * x + 4.0
    } else {
        -1.0
    }
}

fn main() {


}