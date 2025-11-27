//! Command-line interface for llama2-rs.

use clap::{Arg, Command};
use llama2_rs::{Config, Result, Sampler, Tokenizer, Transformer};
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<()> {
    let matches = Command::new("llama2-rs")
        .version("0.1.0")
        .author("AI Assistant")
        .about("Modern Rust implementation of Llama 2 inference")
        .arg(
            Arg::new("checkpoint")
                .help("Path to model checkpoint file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("tokenizer")
                .short('z')
                .long("tokenizer")
                .help("Path to tokenizer file")
                .default_value("tokenizer.bin"),
        )
        .arg(
            Arg::new("temperature")
                .short('t')
                .long("temperature")
                .help("Sampling temperature (0.0 = greedy)")
                .value_parser(clap::value_parser!(f32))
                .default_value("1.0"),
        )
        .arg(
            Arg::new("top_p")
                .short('p')
                .long("top-p")
                .help("Top-p (nucleus) sampling threshold")
                .value_parser(clap::value_parser!(f32))
                .default_value("0.9"),
        )
        .arg(
            Arg::new("steps")
                .short('n')
                .long("steps")
                .help("Maximum number of tokens to generate")
                .value_parser(clap::value_parser!(usize))
                .default_value("256"),
        )
        .arg(
            Arg::new("prompt")
                .short('i')
                .long("prompt")
                .help("Input prompt text"),
        )
        .arg(
            Arg::new("seed")
                .short('s')
                .long("seed")
                .help("Random seed for sampling")
                .value_parser(clap::value_parser!(u64))
                .default_value("0"),
        )
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .help("Mode: generate or chat")
                .value_parser(["generate", "chat"])
                .default_value("generate"),
        )
        .arg(
            Arg::new("system_prompt")
                .short('y')
                .long("system-prompt")
                .help("System prompt for chat mode"),
        )
        .get_matches();

    // Parse arguments
    let checkpoint_path = matches.get_one::<String>("checkpoint").unwrap();
    let tokenizer_path = matches.get_one::<String>("tokenizer").unwrap();
    let temperature = *matches.get_one::<f32>("temperature").unwrap();
    let top_p = *matches.get_one::<f32>("top_p").unwrap();
    let steps = *matches.get_one::<usize>("steps").unwrap();
    let prompt = matches.get_one::<String>("prompt").cloned();
    let seed = *matches.get_one::<u64>("seed").unwrap();
    let mode = matches.get_one::<String>("mode").unwrap();
    let system_prompt = matches.get_one::<String>("system_prompt").cloned();

    // Load model
    println!("Loading model from {}...", checkpoint_path);
    let mut transformer = Transformer::from_checkpoint(checkpoint_path)?;
    println!("Model loaded successfully!");

    // Load tokenizer
    println!("Loading tokenizer from {}...", tokenizer_path);
    let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;
    println!("Tokenizer loaded successfully!");

    // Create sampler
    let mut sampler = Sampler::new(transformer.config.vocab_size, temperature, top_p, seed);

    match mode.as_str() {
        "generate" => {
            run_generation_mode(&mut transformer, &mut tokenizer, &mut sampler, prompt, steps)?;
        }
        "chat" => {
            run_chat_mode(
                &mut transformer,
                &mut tokenizer,
                &mut sampler,
                prompt,
                system_prompt,
                steps,
            )?;
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn run_generation_mode(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<String>,
    steps: usize,
) -> Result<()> {
    // Get prompt from argument or stdin
    let prompt_text = match prompt {
        Some(p) => p,
        None => {
            print!("Enter prompt: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    if prompt_text.is_empty() {
        println!("Empty prompt, exiting.");
        return Ok(());
    }

    println!("Generating with prompt: {}", prompt_text);

    // Generate text
    let generated = transformer.generate(tokenizer, sampler, &prompt_text, steps)?;

    println!("\nGenerated text:");
    println!("{}", generated);

    Ok(())
}

fn run_chat_mode(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    initial_prompt: Option<String>,
    system_prompt: Option<String>,
    steps: usize,
) -> Result<()> {
    println!("Chat mode activated. Type 'quit' or 'exit' to end the conversation.");

    // Handle initial system prompt and user prompt
    let mut conversation_tokens = Vec::new();

    if let Some(sys_prompt) = system_prompt {
        // Format as Llama 2 chat template
        let chat_prompt = format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", sys_prompt);
        let tokens = tokenizer.encode(&chat_prompt, true, false);
        conversation_tokens.extend(tokens);
    }

    if let Some(user_prompt) = initial_prompt {
        let chat_prompt = if system_prompt.is_none() {
            format!("[INST] {} [/INST]", user_prompt)
        } else {
            user_prompt
        };
        let tokens = tokenizer.encode(&chat_prompt, false, false);
        conversation_tokens.extend(tokens);
    }

    // Main chat loop
    let mut pos = 0;
    let mut current_token = if !conversation_tokens.is_empty() {
        conversation_tokens.remove(0)
    } else {
        1 // BOS token
    };

    // Process initial prompts
    for &token in &conversation_tokens {
        transformer.forward(current_token, pos)?;
        pos += 1;
        current_token = token;
    }

    // Interactive chat
    loop {
        // Get user input
        print!("User: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("quit") || user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Format user message for chat
        let chat_message = format!("[INST] {} [/INST]", user_input);
        let message_tokens = tokenizer.encode(&chat_message, false, false);

        // Process user message
        for &token in &message_tokens {
            transformer.forward(current_token, pos)?;
            pos += 1;
            if pos >= steps {
                println!("\nReached maximum steps.");
                return Ok(());
            }
            current_token = token;
        }

        // Generate assistant response
        print!("Assistant: ");
        io::stdout().flush()?;

        let mut response_tokens = Vec::new();
        let mut response_started = false;

        // Generate tokens until EOS or max steps
        while pos < steps {
            let logits = transformer.forward(current_token, pos)?;
            let mut logits_copy = logits.to_vec();

            let next_token = sampler.sample(&mut logits_copy)?;

            if next_token == 2 {
                // EOS token
                break;
            }

            response_tokens.push(next_token);

            // Decode and print token
            if let Ok(piece) = tokenizer.decode(current_token, next_token) {
                let safe_piece = tokenizer.safe_print(&piece);
                print!("{}", safe_piece);
                io::stdout().flush()?;
                response_started = true;
            }

            current_token = next_token;
            pos += 1;
        }

        println!(); // New line after response

        // Reset for next interaction if needed
        if pos >= transformer.config.seq_len {
            println!("Warning: Approaching sequence length limit. Starting fresh conversation.");
            pos = 0;
        }
    }

    Ok(())
}
