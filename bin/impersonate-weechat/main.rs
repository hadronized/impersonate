use impersonate::trainers::weechat::WeechatLogTrainer;
use impersonate::{ChainParameters, LearningParameters, MarkovChainGenerator, Trainer};
use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct CLIOpt {
  path: PathBuf,

  #[structopt(short, long)]
  /// Name of the author to mimick.
  author: Option<String>,

  #[structopt(short, long, default_value = "2")]
  /// Number of words to use to form a wording while learning.
  learning_size: usize,

  #[structopt(short, long, default_value = "1")]
  /// Number of random strings to generate.
  output_strings: usize,

  #[structopt(short = "s", long)]
  /// Number of maximum wordings to use while generating random strings.
  output_size: Option<usize>,
}

fn main() {
  let CLIOpt {
    path,
    author,
    learning_size,
    output_strings,
    output_size,
  } = CLIOpt::from_args();
  let author = author.unwrap_or(String::new());

  let mut markov_chain_generator = MarkovChainGenerator::new();
  let mut trainer = WeechatLogTrainer::new(author, fs::read_to_string(path).unwrap());

  trainer
    .source_train(
      &mut markov_chain_generator,
      LearningParameters {
        wording_size: learning_size,
      },
    )
    .unwrap();

  for _ in 0..output_strings {
    if let Ok(output) = markov_chain_generator.generate_chain(&ChainParameters {
      max_state_traversal: output_size,
    }) {
      println!("{}", output);
    }
  }
}
