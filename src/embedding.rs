// Embedding logic: model loading and vector generation (all-MiniLM-L6-v2, 384 dims).
// Inlined from the former vectorizer crate so vectorizer-napi is standalone.

use anyhow::Result;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

pub use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

pub const MODEL_NAME: &str = "all-MiniLM-L6-v2";

pub fn create_model() -> Result<SentenceEmbeddingsModel> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model()?;
    Ok(model)
}

pub fn generate_embedding(model: &SentenceEmbeddingsModel, text: &str) -> Result<Vec<f32>> {
    let embeddings = model.encode(&[text])?;
    Ok(embeddings[0].clone())
}
