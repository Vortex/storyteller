use futures::{SinkExt, StreamExt};
use reqwest::Error;
use serde_json::{json, Value};
use std::env;
use tungstenite::{connect, Message};
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    match args.get(1) {
        Some(mode) => {
            if mode == "streaming" {
                streaming_mode().await?;
            } else {
                non_streaming_mode().await?;
            }
        }
        None => {
            non_streaming_mode().await?;
        }
    }
    Ok(())
}
async fn non_streaming_mode() -> Result<(), Error> {
    let initial_prompt = "Generate me 5 NPC characters with name, sex, age, and skillsets (use Dwarf Fortress skills):";

    let request = json!({
        "prompt": initial_prompt,
        "max_new_tokens": 250,
        "do_sample": true,
        "temperature": 1.3,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": false,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": true,
        "truncation_length": 2048,
        "ban_eos_token": false,
        "skip_special_tokens": true,
        "stopping_strings": []
    });

    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:5000/api/v1/generate")
        .json(&request)
        .send()
        .await?;

    if response.status().is_success() {
        let response_body: serde_json::Value = response.json().await?;
        let generated_text = response_body["results"][0]["text"]
            .as_str()
            .unwrap_or_default();
        println!("{}{}", initial_prompt, generated_text);

        // Extract the generated NPC characters
        let npc_characters = extract_npc_characters(&generated_text);

        // Generate background stories and relationships for each NPC character
        for npc in npc_characters {
            let character_prompt = format!(
                "Generate a background story and relationships for NPC character {}:",
                npc.name
            );

            let request = json!({
                "prompt": character_prompt,
                // Additional parameters specific to generating background stories and relationships
            });

            let response = client
                .post("http://localhost:5000/api/v1/generate")
                .json(&request)
                .send()
                .await?;

            if response.status().is_success() {
                let response_body: serde_json::Value = response.json().await?;
                let generated_text = response_body["results"][0]["text"]
                    .as_str()
                    .unwrap_or_default();
                println!("{}{}", character_prompt, generated_text);
            } else {
                println!("Request failed with status code: {}", response.status());
            }
        }
    } else {
        println!("Request failed with status code: {}", response.status());
    }

    Ok(())
}

fn extract_npc_characters(generated_text: &str) -> Vec<NpcCharacter> {
    let mut npc_characters = Vec::new();

    // Split the generated text by lines
    let lines: Vec<&str> = generated_text.trim().split('\n').collect();

    for line in lines {
        // Split each line by colon (':') to separate the attribute name and value
        let parts: Vec<&str> = line.split(':').map(|part| part.trim()).collect();

        if parts.len() == 2 {
            let attribute = parts[0];
            let value = parts[1];

            match attribute {
                "Name" => {
                    let name = value.to_string();
                    npc_characters.push(NpcCharacter {
                        name,
                        ..Default::default()
                    });
                }
                "Sex" => {
                    let sex = value.to_string();
                    if let Some(npc) = npc_characters.last_mut() {
                        npc.sex = sex;
                    }
                }
                "Age" => {
                    if let Ok(age) = value.parse::<u32>() {
                        if let Some(npc) = npc_characters.last_mut() {
                            npc.age = age;
                        }
                    }
                }
                "Skillset" => {
                    let skillset = value.to_string();
                    if let Some(npc) = npc_characters.last_mut() {
                        npc.skillset = skillset;
                    }
                }
                _ => (),
            }
        }
    }

    npc_characters
}

#[derive(Default)]
struct NpcCharacter {
    name: String,
    sex: String,
    age: u32,
    skillset: String,
}
async fn streaming_mode() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "In order to make homemade bread, follow these steps:\n1)";

    let (mut socket, response) = connect(Url::parse("ws://localhost:5005/api/v1/stream").unwrap())
        .expect("Failed to connect");

    println!("Connected to the server");
    println!("Response HTTP code: {}", response.status());
    println!("Response contains the following headers:");
    for (header, _) in response.headers() {
        println!("* {}", header);
    }

    println!("Connected to the server");
    let request = json!({
        "prompt": prompt,
        "max_new_tokens": 250,
        "do_sample": true,
        "temperature": 1.3,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": false,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": true,
        "truncation_length": 2048,
        "ban_eos_token": false,
        "skip_special_tokens": true,
        "stopping_strings": []    });

    let msg = Message::Text(request.to_string());
    socket.write_message(msg).unwrap();

    loop {
        let msg = socket.read_message().expect("Error reading message");
        match msg {
            Message::Text(text) => {
                let incoming_data: serde_json::Value = serde_json::from_str(&text).unwrap();
                match incoming_data["event"].as_str() {
                    Some("text_stream") => println!("{}", incoming_data["text"].as_str().unwrap()),
                    Some("stream_end") => break,
                    _ => (),
                }
            }
            _ => (),
        }
    }

    Ok(())
}
