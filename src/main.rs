use std::thread::sleep;
use std::time::Duration;

mod activation_functions;
mod neuron;
mod dqn;
mod vectors;
mod fitting;
mod loss_functions;

fn main() {
    println!("Hello, world!");
    loop {
        let input = TerminalInput::get_input();
        if !input.is_empty() {
            println!("{:?}", input);
        }
        let input = TerminalInput::get_input();
        if !input.is_empty() {
            println!("{:?}", input);
        }
        sleep(Duration::from_millis(60));
    }
}
