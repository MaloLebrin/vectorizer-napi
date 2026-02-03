// Addon NAPI-RS : expose vectorize(text) -> Promise<Float64Array>
// Le modèle rust-bert (tch) n'est pas Sync ; on utilise un modèle par thread (thread_local)
// dans le pool libuv pour ne pas bloquer le main thread.
// Voir https://napi.rs/docs/concepts/async-task

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::cell::RefCell;

mod embedding;
use embedding::{create_model, generate_embedding, MODEL_NAME};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

// Model per thread: each libuv pool worker loads the model on first use.
thread_local! {
    static MODEL: RefCell<Option<SentenceEmbeddingsModel>> = RefCell::new(None);
}

fn get_or_init_model() -> napi::Result<()> {
    MODEL.with(|m| {
        if m.borrow().is_none() {
            let model = create_model().map_err(|e| {
                napi::Error::from_reason(format!("Failed to load embedding model: {}", e))
            })?;
            *m.borrow_mut() = Some(model);
        }
        Ok(())
    })
}

/// Tâche asynchrone : calcul de l'embedding dans le thread pool.
pub struct VectorizeTask {
    text: String,
}

#[napi]
impl Task for VectorizeTask {
    type Output = Vec<f32>;
    type JsValue = Float64Array;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        get_or_init_model()?;
        MODEL.with(|m| {
            let model = m.borrow();
            let model = model.as_ref().expect("model initialized");
            generate_embedding(model, &self.text)
                .map_err(|e| napi::Error::from_reason(format!("Embedding failed: {}", e)))
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        let f64_vec: Vec<f64> = output.into_iter().map(|x| x as f64).collect();
        Ok(f64_vec.into())
    }
}

/// Vectorise un texte et retourne un vecteur de 384 dimensions (all-MiniLM-L6-v2).
/// S'exécute dans le thread pool sans bloquer le main thread Node.js.
#[napi]
pub fn vectorize(text: String) -> AsyncTask<VectorizeTask> {
    AsyncTask::new(VectorizeTask { text })
}

/// Nom du modèle utilisé (pour traçabilité).
#[napi]
pub fn model_name() -> String {
    MODEL_NAME.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_returns_expected_string() {
        assert_eq!(model_name(), "all-MiniLM-L6-v2");
    }

    #[test]
    fn embedding_conversion_f32_to_f64_preserves_values() {
        let f32_vec = vec![1.0f32, 2.5f32, -0.5f32];
        let f64_vec: Vec<f64> = f32_vec.iter().map(|x| *x as f64).collect();
        assert_eq!(f64_vec.len(), 3);
        assert!((f64_vec[0] - 1.0).abs() < 1e-10);
        assert!((f64_vec[1] - 2.5).abs() < 1e-10);
        assert!((f64_vec[2] - (-0.5)).abs() < 1e-10);
    }
}
