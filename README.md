# Storyteller: NPC Generator

> :warning: **Disclaimer**: This project is currently in a very early stage of development. It may lack certain features, and there may be bugs. Please don't use it.


Storyteller is a library that generates non-player characters (NPCs) for your game. It uses a language model to create characters and their backstories, which can then be utilized in your game to create a rich, immersive experience.

## Features

- **Generate NPCs**: Create unique NPCs with distinct names, genders, ages, and skill sets.
- **Detailed Backstories**: Each NPC comes with a backstory and relationships, providing depth to their character.
- **JSON Validation**: Generated NPCs are returned in a validated JSON format, ensuring the information is consistent and accurate.
- **Database Integration**: All NPCs are stored in a database for easy retrieval and management.

## Installation

TODO: Add installation instructions here. You might want to include instructions for building from source, and also for installing any precompiled binaries (if available).

## Usage

To use this library in your Rust project, add the following line to your `Cargo.toml` under `[dependencies]`:

```toml
storyteller = "0.1.0" # Replace with the actual version number
```

Then, in your Rust files, you can use the library like this:

```rust
extern crate storyteller;

async fn main() {
    let npc = storyteller::generate_npc().await.unwrap();
    println!("{:?}", npc);
}
```

## Database Setup

TODO: Add instructions for setting up the database that Storyteller uses

## Configuration

TODO: If there are any configurations that users can adjust, document them here.

