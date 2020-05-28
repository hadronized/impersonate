//! A [`Trainer`] that can learn from a Weechat log.

use lazy_static::lazy_static;
use regex::Regex;

use crate::{ChainError, LearningParameters, MarkovChainGenerator, Trainer};

lazy_static! {
  static ref REGEX_LINE: Regex =
    Regex::new(r"(^\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2}: (.*)$").unwrap();
}

/// The content of a weechat log.
pub struct WeechatLog {
  lines: Vec<String>,
  /// The author we are interested in
  author: String,
}

impl WeechatLog {
  pub fn new(content: String, author: String) -> Self {
    let lines = content
      .split_terminator('\n')
      .map(|line| line.to_owned())
      .collect();

    Self { lines, author }
  }

  /// Filter the log by removing all the noise linked to Weechat.
  fn filter_noise(&mut self) {
    self.lines.retain(|line| {
      if let Some(captures) = REGEX_LINE.captures(line) {
        !(captures[2].starts_with("--")
          || captures[2].starts_with("<--")
          || captures[2].starts_with("-->"))
      } else {
        true
      }
    });
  }

  /// Clean up lines to remove dates and nicknames.
  fn cleanup(&mut self) {
    let author_start = format!("{} ", self.author);

    for line in &mut self.lines {
      // remove the date
      if let Some(captures) = REGEX_LINE.captures(line) {
        let input = &captures[2];

        if input.starts_with(&author_start) {
          // remove the nickname
          let content = input[author_start.len()..].to_owned();

          *line = content;
        } else {
          // set the line to the empty line so that we drop it
          *line = String::new();
        }
      }
    }

    // remove empty lines
    self.lines.retain(|line| !line.is_empty());
  }
}

impl Trainer for WeechatLog {
  fn source_train(
    &mut self,
    markov_chain_generator: &mut MarkovChainGenerator,
    learn_params: LearningParameters,
  ) -> Result<(), ChainError> {
    self.filter_noise();
    self.cleanup();

    for line in &self.lines {
      markov_chain_generator.train(&learn_params, line);
    }

    Ok(())
  }
}
