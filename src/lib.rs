use itertools::Itertools as _;
use rand::{thread_rng, Rng as _};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::iter::FromIterator;

/// The smallest amount of wording that can be used to represent Markov
/// states.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Wording {
  /// Parts of the words forming the wording.
  words: Vec<String>,
}

/// Create a wording based on an iterator.
impl FromIterator<String> for Wording {
  fn from_iter<T>(iter: T) -> Self
  where
    T: IntoIterator<Item = String>,
  {
    Self {
      words: iter.into_iter().collect(),
    }
  }
}

impl fmt::Display for Wording {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    if !self.words.is_empty() {
      f.write_str(&self.words[0])?;
    }

    for w in &self.words[1..] {
      write!(f, " {}", w)?;
    }

    Ok(())
  }
}

/// An associated [`Wording`] to another [`Wording`], attached with a number of occurrences it
/// was found as a next wording.
///
/// Examples:
///
/// ```ignore
///   "foo bar zoo" "quux meh"
/// ```
///
/// `"quux meh"` appears `1` time after `"foo bar zoo"` here.
///
/// This type also serves as “arc” in the Markov graph.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Transition {
  count: usize,
}

impl Default for Transition {
  fn default() -> Self {
    Self { count: 0 }
  }
}

/// A set of Markov transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct State {
  nexts: HashMap<Wording, Transition>,
}

impl Default for State {
  fn default() -> Self {
    Self {
      nexts: HashMap::default(),
    }
  }
}

/// A set of Markov states.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkovChainGenerator {
  states: HashMap<Wording, State>,
}

impl MarkovChainGenerator {
  /// Create a new empty Markov chain generator.
  pub fn new() -> Self {
    Self {
      states: HashMap::new(),
    }
  }

  /// Split a line into a chunk of [`Wording`].
  fn chunk_line<L>(learn_param: &LearningParameters, line: L) -> Vec<Wording>
  where
    L: AsRef<str>,
  {
    let LearningParameters { wording_size } = *learn_param;
    let words = line.as_ref().split(' ').map(|line| line.to_owned());

    words
      .chunks(wording_size)
      .into_iter()
      .map(|chunk| chunk.into_iter().collect::<Wording>())
      .collect()
  }

  /// Cut an input string into a set of [`Wording`] and train the generator on it.
  ///
  /// The input parameter tells how the cut will be done.
  pub fn train<L>(&mut self, learn_param: &LearningParameters, line: L)
  where
    L: AsRef<str>,
  {
    let chunks = Self::chunk_line(learn_param, line);

    for (wording1, wording2) in chunks.into_iter().tuple_windows() {
      let state = self.states.entry(wording1).or_insert(State::default());

      state.nexts.entry(wording2).or_default().count += 1;
    }
  }

  /// Generate a random chain.
  pub fn generate_chain(&self, chain_param: &ChainParameters) -> Result<String, ChainError> {
    let mut output = String::new();
    let mut rng = thread_rng();
    let ChainParameters {
      max_state_traversal,
    } = *chain_param;

    // get the initial state
    let ri = rng.gen_range(0, self.states.len());
    let mut key = self
      .states
      .keys()
      .nth(ri)
      .ok_or_else(|| ChainError::TooFewInitialStates(self.states.len()))?;

    eprintln!("initial state is {}", key);
    output = key.to_string();

    for _ in 0..max_state_traversal.unwrap_or(usize::max_value()) {
      if let Some(state) = self.states.get(&key) {
        let mut states = state.nexts.iter().collect::<Vec<_>>();
        states.sort_by(|(_, a), (_, b)| a.count.cmp(&b.count));

        // find the next state to jump to; for this, we need to sort the state by occurrences
        let ri = rng.gen_range(0, states.len());
        key = &states[ri].0;

        write!(&mut output, " {}", key.to_string());
      } else {
        break;
      }
    }

    Ok(output)
  }
}

/// Learning parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LearningParameters {
  /// Size (in words) of wordings to learn.
  ///
  /// Minimum value is `1` and is will generate sentences that makes very little sense. Higher
  /// values will generate more sense but a too high value will make the Markov states “poor”.
  wording_size: usize,
}

/// Chain generation parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainParameters {
  /// Number of states to go through at maximum.
  max_state_traversal: Option<usize>,
}

/// Chain generation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChainError {
  TooFewInitialStates(usize),
}

/// A way to train a Markov chain generator based on a source.
///
/// This trait allows to adapt the way a Markov chain generator can learn without having to know
/// the format of the input source.
pub trait Trainer {
  /// Adapt to the source and train the input [`MarkovChainGenerator`].
  fn source_train(
    &self,
    markov_chain_generator: &mut MarkovChainGenerator,
  ) -> Result<(), ChainError>;
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_simple() {
    let mut generator = MarkovChainGenerator::new();
    let learn_param = LearningParameters { wording_size: 3 };
    let chain_param = ChainParameters {
      max_state_traversal: None,
    };

    generator.train(&learn_param, "foo bar zoo quux hello, world!");
    let result = generator.generate_chain(&chain_param);

    eprintln!("result: {:?}", result);
  }
}
