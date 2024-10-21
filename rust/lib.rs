#[derive(Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct BasicBlock<T>(pub Vec<T>);

impl<T> BasicBlock<T> {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}
