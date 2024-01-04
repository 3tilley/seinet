use std::thread::sleep;
use std::time::Duration;
use crate::driving::input::TerminalInput;
use crate::driving::input::Input;

mod activation_functions;
mod neuron;
mod driving;

fn main() {
    println!("Hello, world!");
    loop {
        let input = TerminalInput::get_input();
        println!("{:?}", input);
        sleep(Duration::from_millis(60));
    }
}