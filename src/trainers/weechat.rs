//! A [`Trainer`] that can learn from a Weechat log.

use lazy_static::lazy_static;
use regex::Regex;

use crate::{ChainError, LearningParameters, MarkovChainGenerator, Trainer};

lazy_static! {
  static ref REGEX_LINE: Regex =
    Regex::new(r"(\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2}:\d{2}\s+(.*)").unwrap();
}

/// The content of a weechat log.
pub struct WeechatLogTrainer {
  lines: Vec<String>,
  /// The author we are interested in
  author: String,
}

impl WeechatLogTrainer {
  pub fn new(author: impl Into<String>, content: impl Into<String>) -> Self {
    let lines: Vec<_> = content
      .into()
      .split_terminator('\n')
      .map(|line| line.to_owned())
      .collect();
    let author = author.into();

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
        false
      }
    });
  }

  /// Clean up lines to remove dates and nicknames.
  fn cleanup(&mut self) {
    for line in &mut self.lines {
      // remove the date
      if let Some(captures) = REGEX_LINE.captures(line) {
        let mut input = &captures[2];

        if !input.is_empty() && input.as_bytes()[0] == b'@' {
          input = &input[1..];
        }

        if input.starts_with(&self.author) {
          // remove the nickname
          let content = input[self.author.len()..].trim().to_owned();
          eprintln!("{}", content);

          *line = content;
        } else {
          // set the line to the empty line so that we drop it
          eprintln!("\tignoring {}", input);
          *line = String::new();
        }
      }
    }

    // remove empty lines
    self.lines.retain(|line| !line.is_empty());
  }
}

impl Trainer for WeechatLogTrainer {
  fn source_train(
    &mut self,
    markov_chain_generator: &mut MarkovChainGenerator,
    learn_params: LearningParameters,
  ) -> Result<(), ChainError> {
    self.filter_noise();
    self.cleanup();

    eprintln!("learning from Weechat log ({} lines)", self.lines.len());
    for line in &self.lines {
      markov_chain_generator.train(&learn_params, line);
    }

    Ok(())
  }
}
