use crate::store::Store;
use crate::temp_store::TempStore;
use crate::chunker::Chunker;
use crate::embedder::Embedder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum ChatMessage {
    System(String),
    User(String),
    Assistant(AssistantMessageContent),
    Tool(ToolMessageContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessageContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessageContent {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Thought(String),
    ToolCall { name: String, args: serde_json::Value },
    ToolResult { name: String, result: String },
    Answer(String),
    Status(String),
    Error(String),
}

pub struct GhostAgent {
    store: Arc<Store>,
    temp_store: Arc<TempStore>,
    chunker: Arc<Chunker>,
    embedder: Arc<Embedder>,
    client: reqwest::Client,
}

impl GhostAgent {
    pub fn new(store: Arc<Store>, temp_store: Arc<TempStore>, chunker: Arc<Chunker>, embedder: Arc<Embedder>) -> Self {
        Self {
            store,
            temp_store,
            chunker,
            embedder,
            client: reqwest::Client::new(),
        }
    }

    /// Primary Turn Loop for the Agent
    pub async fn run_loop(
        &self,
        session_id: String,
        user_message: String,
        tx: mpsc::UnboundedSender<Result<axum::response::sse::Event, std::convert::Infallible>>,
    ) {
        let emit = |event: AgentEvent| {
            if let Ok(json) = serde_json::to_string(&event) {
                let sse_event = axum::response::sse::Event::default().event("agent").data(json);
                let _ = tx.send(Ok(sse_event));
            }
        };

        emit(AgentEvent::Status("Recalling memories and preparing workspace...".to_string()));

        // Step 1: Memory Recall
        let query_embedding = self.embedder.embed(&user_message).ok();
        let recall_context = match self.store.search_hybrid(
            &session_id,
            &user_message,
            query_embedding.as_deref(),
            3,
            None,
            168.0,
            "organize",
            &[],
            &lume_hybrid::bm25::Bm25Params::default(),
        ) {
            Ok(results) => {
                if results.is_empty() {
                    "No relevant memories recalled.".to_string()
                } else {
                    let mut s = String::from("Recalled Memories:\n");
                    for (i, (chunk, score)) in results.iter().enumerate() {
                        s.push_str(&format!("{}. [{}] (score: {:.4}) {}\n", i + 1, chunk.id, score, chunk.text));
                    }
                    s
                }
            }
            Err(e) => format!("Failed to recall memories: {}", e),
        };

        // Step 2: System Prompt & History Setup
        let system_prompt = format!(
            "You are the shivvr cognitive AI Agent, designed for high-performance semantic operations.\n\
             You are running inside a Linux Docker container environment.\n\
             You have access to memory retrieval, storage, and local command execution tools.\n\n\
             {}\n\n\
             Guidelines:\n\
             - Use the tools provided to achieve the user's goal.\n\
             - Inspect the session memories to maintain context.\n\
             - If the user asks you to run a command, use `run_command`.\n\
             - Provide a clear, final summary when done.",
            recall_context
        );

        let mut history = vec![
            ChatMessage::System(system_prompt),
            ChatMessage::User(user_message),
        ];

        let max_turns = 10;
        let mut turn = 0;

        let anthropic_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let openai_key = std::env::var("OPENAI_API_KEY").ok();

        if anthropic_key.is_none() && openai_key.is_none() {
            emit(AgentEvent::Error("No LLM API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY) detected. Falling back to local simulator.".to_string()));
            // Fast simulation for demonstration/testing
            emit(AgentEvent::Thought("I see you wanted to test my cognitive capabilities! Since no LLM API key was provided, I am running in simulation mode.".to_string()));
            emit(AgentEvent::ToolCall {
                name: "search_memory".to_string(),
                args: serde_json::json!({ "session_id": session_id, "query": "test query" }),
            });
            emit(AgentEvent::ToolResult {
                name: "search_memory".to_string(),
                result: recall_context.clone(),
            });
            emit(AgentEvent::Answer("This is a simulated final agent answer. Expose your OPENAI_API_KEY or ANTHROPIC_API_KEY to run fully autonomous turn loops!".to_string()));
            return;
        }

        while turn < max_turns {
            turn += 1;
            emit(AgentEvent::Status(format!("Agent turn {}/{} starting...", turn, max_turns)));

            // Request completions from Anthropic or OpenAI
            let llm_res = if let Some(ref ant_key) = anthropic_key {
                self.call_anthropic(ant_key, &history).await
            } else if let Some(ref oai_key) = openai_key {
                self.call_openai(oai_key, &history).await
            } else {
                Err(anyhow::anyhow!("No API key available"))
            };

            let assistant_msg = match llm_res {
                Ok(msg) => msg,
                Err(e) => {
                    emit(AgentEvent::Error(format!("LLM invocation error: {}", e)));
                    return;
                }
            };

            // Stream assistant thoughts
            if let Some(ref text) = assistant_msg.text {
                emit(AgentEvent::Thought(text.clone()));
            }

            // Check for tool calls
            if let Some(ref tool_calls) = assistant_msg.tool_calls {
                if tool_calls.is_empty() {
                    // No tool calls - this is the final answer!
                    if let Some(ref text) = assistant_msg.text {
                        emit(AgentEvent::Answer(text.clone()));
                    }
                    break;
                }

                // Add assistant response to history
                history.push(ChatMessage::Assistant(assistant_msg.clone()));

                // Execute tool calls
                for tc in tool_calls {
                    let args_parsed: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                    emit(AgentEvent::ToolCall {
                        name: tc.name.clone(),
                        args: args_parsed.clone(),
                    });

                    emit(AgentEvent::Status(format!("Executing tool {}...", tc.name)));
                    let tool_result = self.execute_tool(&tc.name, args_parsed).await;
                    emit(AgentEvent::ToolResult {
                        name: tc.name.clone(),
                        result: tool_result.clone(),
                    });

                    // Add tool response to history
                    history.push(ChatMessage::Tool(ToolMessageContent {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        content: tool_result,
                    }));
                }
            } else {
                // No tool calls at all
                if let Some(ref text) = assistant_msg.text {
                    emit(AgentEvent::Answer(text.clone()));
                }
                break;
            }
        }

        if turn >= max_turns {
            emit(AgentEvent::Error("Max agent reasoning turns exceeded.".to_string()));
        }
    }

    /// Tool Execution Router
    async fn execute_tool(&self, name: &str, args: serde_json::Value) -> String {
        match name {
            "search_memory" => {
                let session_id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("default");
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let lexical_only = args.get("lexical_only").and_then(|v| v.as_bool()).unwrap_or(false);

                let query_embedding = if lexical_only {
                    None
                } else {
                    self.embedder.embed(query).ok()
                };

                match self.store.search_hybrid(
                    session_id,
                    query,
                    query_embedding.as_deref(),
                    5,
                    None,
                    168.0,
                    "organize",
                    &[],
                    &lume_hybrid::bm25::Bm25Params::default(),
                ) {
                    Ok(results) => {
                        let mapped: Vec<serde_json::Value> = results.into_iter().map(|(chunk, score)| {
                            serde_json::json!({
                                "chunk_id": chunk.id,
                                "score": score,
                                "text": chunk.text,
                                "source": chunk.source,
                                "metadata": chunk.metadata,
                                "created_at": chunk.created_at.to_rfc3339()
                            })
                        }).collect();
                        serde_json::to_string_pretty(&mapped).unwrap_or_else(|_| "[]".to_string())
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "ingest_memory" => {
                let session_id = args.get("session_id").and_then(|v| v.as_str()).unwrap_or("default");
                let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let source = args.get("source").and_then(|v| v.as_str()).map(|s| s.to_string());

                match self.chunker.chunk(text, source, serde_json::Value::Null).await {
                    Ok(chunks) => {
                        if chunks.is_empty() {
                            "Success: Text was too short or no chunks produced.".to_string()
                        } else {
                            let first_id = chunks[0].id.clone();
                            match self.store.add_chunks(session_id, chunks, None) {
                                Ok(_) => format!("Success: Memory chunk indexed under ID {}", first_id),
                                Err(e) => format!("Error indexing chunk in store: {}", e),
                            }
                        }
                    }
                    Err(e) => format!("Error chunking text: {}", e),
                }
            }
            "list_sessions" => {
                let sessions = self.store.list_sessions(None).unwrap_or_default();
                let mapped: Vec<serde_json::Value> = sessions.into_iter().map(|s| {
                    serde_json::json!({
                        "id": s.id,
                        "chunks": s.chunk_count,
                        "last_active": s.last_ingested.to_rfc3339()
                    })
                }).collect();
                serde_json::to_string_pretty(&mapped).unwrap_or_else(|_| "[]".to_string())
            }
            "run_command" => {
                let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                if command.is_empty() {
                    return "Error: Empty command".to_string();
                }
                
                #[cfg(unix)]
                {
                    match std::process::Command::new("sh")
                        .arg("-c")
                        .arg(command)
                        .output()
                    {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                            format!("Exit Code: {}\nSTDOUT:\n{}\nSTDERR:\n{}", output.status.code().unwrap_or(-1), stdout, stderr)
                        }
                        Err(e) => format!("Error executing command: {}", e),
                    }
                }
                #[cfg(not(unix))]
                {
                    "Error: run_command is only supported in Unix/Linux container environments".to_string()
                }
            }
            _ => format!("Error: Tool '{}' not found", name),
        }
    }

    /// OpenAI API Call Handler
    async fn call_openai(&self, key: &str, history: &[ChatMessage]) -> anyhow::Result<AssistantMessageContent> {
        let endpoint = std::env::var("OPENAI_API_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
        
        let oai_model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let mut messages = Vec::new();
        for msg in history {
            match msg {
                ChatMessage::System(content) => {
                    messages.push(serde_json::json!({ "role": "system", "content": content }));
                }
                ChatMessage::User(content) => {
                    messages.push(serde_json::json!({ "role": "user", "content": content }));
                }
                ChatMessage::Assistant(content) => {
                    let mut m = serde_json::json!({ "role": "assistant" });
                    if let Some(ref text) = content.text {
                        m["content"] = serde_json::json!(text);
                    }
                    if let Some(ref tcs) = content.tool_calls {
                        let mut tc_list = Vec::new();
                        for tc in tcs {
                            tc_list.push(serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": { "name": tc.name, "arguments": tc.arguments }
                            }));
                        }
                        m["tool_calls"] = serde_json::json!(tc_list);
                    }
                    messages.push(m);
                }
                ChatMessage::Tool(content) => {
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": content.tool_call_id,
                        "name": content.name,
                        "content": content.content
                    }));
                }
            }
        }

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Query high-performance hybrid memory search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" },
                            "query": { "type": "string" },
                            "lexical_only": { "type": "boolean" }
                        },
                        "required": ["session_id", "query"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "ingest_memory",
                    "description": "Store a new memory fragment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" },
                            "text": { "type": "string" },
                            "source": { "type": "string" }
                        },
                        "required": ["session_id", "text"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "list_sessions",
                    "description": "List all active indexed memory sessions.",
                    "parameters": { "type": "object", "properties": {} }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command inside the docker system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": { "type": "string", "description": "The command string to execute." }
                        },
                        "required": ["command"]
                    }
                }
            }),
        ];

        let payload = serde_json::json!({
            "model": oai_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        });

        let res = self.client.post(&endpoint)
            .bearer_auth(key)
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let body = res.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", body));
        }

        let resp_json: serde_json::Value = res.json().await?;
        let choice = resp_json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .ok_or_else(|| anyhow::anyhow!("Failed to parse OpenAI message content"))?;

        let text = choice.get("content").and_then(|v| v.as_str()).map(|s| s.to_string());
        
        let tool_calls = choice.get("tool_calls").and_then(|tcs| {
            tcs.as_array().map(|arr| {
                arr.iter().filter_map(|val| {
                    let id = val.get("id")?.as_str()?.to_string();
                    let name = val.get("function")?.get("name")?.as_str()?.to_string();
                    let arguments = val.get("function")?.get("arguments")?.as_str()?.to_string();
                    Some(ToolCall { id, name, arguments })
                }).collect::<Vec<_>>()
            })
        });

        Ok(AssistantMessageContent { text, tool_calls })
    }

    /// Anthropic Messages API Call Handler
    async fn call_anthropic(&self, key: &str, history: &[ChatMessage]) -> anyhow::Result<AssistantMessageContent> {
        let endpoint = "https://api.anthropic.com/v1/messages";
        let ant_model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-haiku-20241022".to_string());

        let mut system = None;
        let mut messages = Vec::new();

        for msg in history {
            match msg {
                ChatMessage::System(content) => {
                    system = Some(content.clone());
                }
                ChatMessage::User(content) => {
                    messages.push(serde_json::json!({ "role": "user", "content": content }));
                }
                ChatMessage::Assistant(content) => {
                    let mut contents = Vec::new();
                    if let Some(ref text) = content.text {
                        contents.push(serde_json::json!({ "type": "text", "text": text }));
                    }
                    if let Some(ref tcs) = content.tool_calls {
                        for tc in tcs {
                            let input: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                            contents.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": input
                            }));
                        }
                    }
                    messages.push(serde_json::json!({ "role": "assistant", "content": contents }));
                }
                ChatMessage::Tool(content) => {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.tool_call_id,
                                "content": content.content
                            }
                        ]
                    }));
                }
            }
        }

        let tools = vec![
            serde_json::json!({
                "name": "search_memory",
                "description": "Query hybrid dense/sparse vector memory store.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "query": { "type": "string" },
                        "lexical_only": { "type": "boolean" }
                    },
                    "required": ["session_id", "query"]
                }
            }),
            serde_json::json!({
                "name": "ingest_memory",
                "description": "Index a new memory string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "text": { "type": "string" },
                        "source": { "type": "string" }
                    },
                    "required": ["session_id", "text"]
                }
            }),
            serde_json::json!({
                "name": "list_sessions",
                "description": "Return active memory sessions.",
                "input_schema": { "type": "object", "properties": {} }
            }),
            serde_json::json!({
                "name": "run_command",
                "description": "Execute a shell command inside the docker.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "The command string to execute." }
                    },
                    "required": ["command"]
                }
            }),
        ];

        let mut payload = serde_json::json!({
            "model": ant_model,
            "max_tokens": 1024,
            "messages": messages,
            "tools": tools,
        });

        if let Some(sys) = system {
            payload["system"] = serde_json::json!(sys);
        }

        let res = self.client.post(endpoint)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let body = res.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", body));
        }

        let resp_json: serde_json::Value = res.json().await?;
        let content_arr = resp_json.get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| anyhow::anyhow!("Failed to parse Anthropic message content array"))?;

        let mut text = None;
        let mut tool_calls = Vec::new();

        for block in content_arr {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if block_type == "text" {
                text = block.get("text").and_then(|t| t.as_str()).map(|s| s.to_string());
            } else if block_type == "tool_use" {
                let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let input = block.get("input").cloned().unwrap_or(serde_json::Value::Null);
                let arguments = serde_json::to_string(&input).unwrap_or_default();
                tool_calls.push(ToolCall { id, name, arguments });
            }
        }

        let tcs = if tool_calls.is_empty() { None } else { Some(tool_calls) };

        Ok(AssistantMessageContent { text, tool_calls: tcs })
    }
}
