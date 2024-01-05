use std::time::Duration;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, poll, read};
use crossterm::event::Event::Key;
use crate::driving::input::Accelerator::{Accelerate, Brake};

#[derive(Debug, Copy, Clone)]
pub enum Accelerator {
    Accelerate,
    Brake,
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct KeyInput {
    pub acceleration: Option<Accelerator>,
    pub direction: Option<Direction>,
    pub key_state: Option<KeyEventKind>,
}

impl KeyInput {
    pub fn new(acceleration: Option<Accelerator>, direction: Option<Direction>, key_state: Option<KeyEventKind>) -> KeyInput {
        KeyInput {
            acceleration,
            direction,
            key_state,
        }
    }
     pub fn is_empty(&self) -> bool {
         self.acceleration.is_none() && self.direction.is_none()
     }
}


pub trait Input {
    fn get_input() -> KeyInput;
}

pub struct TerminalInput {


}

impl Input for TerminalInput {

    fn get_input() -> KeyInput {
        // let stream = crossterm::event::E

        if poll(Duration::from_micros(10)).unwrap() {
            // It's guaranteed that `read` won't block, because `poll` returned
            // `Ok(true)`.
            let event = read().unwrap();
            if let Event::Key(key) = event {
                match key {
                    KeyEvent { code, modifiers, kind, state } => {
                        if kind != KeyEventKind::Release {
                            match code {
                                KeyCode::Left => return KeyInput::new(None, Some(Direction::Left), Some(kind)),
                                KeyCode::Right => return KeyInput::new(None, Some(Direction::Right), Some(kind)),
                                KeyCode::Up => return KeyInput::new(Some(Accelerate), None, Some(kind)),
                                KeyCode::Down => return KeyInput::new(Some(Brake), None, Some(kind)),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        KeyInput::default()
    }
}