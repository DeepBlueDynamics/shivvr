use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Terminal,
};
use std::{
    collections::VecDeque,
    fs,
    io::{self, BufRead, Read, Stdout, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};
use serde::{Deserialize, Serialize};
use unicode_width::UnicodeWidthChar;

const NEGOTIATE_MIN_TOKENS: usize = 4;
const NEGOTIATE_MIN_WORDS: usize = NEGOTIATE_MIN_TOKENS * 2;
const MENU_ITEMS: [&str; 3] = ["Negotiate boundaries", "Show similar", "Negotiate all"];

#[derive(Clone)]
struct ChunkInfo {
    index: usize,
    start: usize,
    end: usize,
    score: f64,
    is_boundary: bool,
    preview: String,
    text: String,
}

struct ChunkSeed {
    start: usize,
    end: usize,
    score: f64,
    text: String,
}

struct SeedAnalysis {
    chunks: Vec<ChunkInfo>,
    boundary_cutoff: f64,
}

#[derive(Clone)]
struct Inversion {
    index: usize,
    original: String,
    inverted: String,
    score: f64,
}

// Persistent model server for caching vec2text model
struct ModelServer {
    stdin: ChildStdin,
    response_rx: Receiver<ModelResponse>,
    #[allow(dead_code)]
    child: Child,
}

struct NegotiateProcess {
    rx: Receiver<NegotiateEvent>,
    child: Child,
}

struct NegotiateBatch {
    queue: VecDeque<usize>,
    total: usize,
    skipped: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum ModelResponse {
    Log { message: String },
    Ready,
    Pong { id: Option<u64> },
    Result {
        id: Option<u64>,
        command: String,
        #[serde(default)]
        embeddings: Vec<Vec<f32>>,
        #[serde(default)]
        inversions: Vec<String>,
        #[serde(default)]
        scores: Vec<f64>,
    },
    Error { id: Option<u64>, message: String },
    Shutdown { id: Option<u64> },
}

#[derive(Serialize)]
struct ModelCommand {
    command: String,
    id: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    texts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_length: Option<u32>,
}

struct App {
    input_path: Option<PathBuf>,
    text: String,
    chunks: Vec<ChunkInfo>,
    selected: usize,
    list_state: ListState,
    window_tokens: usize,
    boundary_cutoff: f64,
    boundary_count: usize,
    status: String,
    score_config: ScoreConfig,
    score_source: String,
    processing: bool,
    processing_pending: bool,
    auto_embed_pending: bool,
    process_running: bool,
    processed_count: usize,
    active_index: Option<usize>,
    highlight_scores: Vec<f32>,
    last_step: Instant,
    flash_until: Option<Instant>,
    embeddings: Vec<Option<Vec<f32>>>,
    pending_embeddings: VecDeque<(usize, Vec<f32>)>,
    logs: VecDeque<String>,
    embed_started_at: Option<Instant>,
    last_embed_log: Option<Instant>,
    embed_done_pending: bool,
    last_embed_tick: Instant,
    // Inversion mode
    invert_running: bool,
    inversions: Vec<Option<Inversion>>,
    show_inversions: bool,
    // Monte Carlo exploration
    explore_running: bool,
    explored_spans: Vec<ScoredSpan>,
    explore_sample_count: usize,
    // GPU throughput tracking
    gpu_ops: VecDeque<Instant>,  // timestamps of recent GPU operations
    gpu_ops_per_sec: f32,        // calculated ops/sec
    // Cooperative negotiation
    negotiate_running: bool,
    negotiate_round: usize,
    negotiate_boundary: Option<usize>,
    negotiate_batch: Option<NegotiateBatch>,
    // Similarity view
    show_similarity: bool,
    similar_chunks: Vec<(usize, f32)>,  // (chunk_index, similarity_score)
    similarity_scroll: usize,
    similarity_selected: usize,  // which item is selected in similarity view
    focus_left: bool,            // true = left panel (similarity) has focus
    expanded_chunk: Option<usize>,  // if Some, show this chunk expanded full-width
    // Source overview scroll
    source_scroll: usize,
    // Cached model server
    model_server: Option<ModelServer>,
    model_request_id: u64,
    // Chunk action menu
    show_menu: bool,
    menu_selection: usize,
    // Similarity dimming (in-place, no separate view)
    dim_dissimilar: bool,
    // Search
    search_query: String,
    search_matches: Vec<(usize, usize)>,  // (start, end) positions in text
    // Focus management
    focus_area: FocusArea,
}

#[derive(Clone, Copy, PartialEq)]
enum FocusArea {
    Search,
    SourceText,
    ChunkList,
}

impl App {
    fn new(input_path: Option<PathBuf>, window_tokens: usize, score_config: ScoreConfig) -> Result<Self> {
        let mut app = Self {
            input_path,
            text: String::new(),
            chunks: Vec::new(),
            selected: 0,
            list_state: ListState::default(),
            window_tokens,
            boundary_cutoff: 0.0,
            boundary_count: 0,
            status: String::new(),
            score_config,
            score_source: "seed".to_string(),
            processing: false,
            processing_pending: false,
            auto_embed_pending: false,
            process_running: false,
            processed_count: 0,
            active_index: None,
            highlight_scores: Vec::new(),
            last_step: Instant::now(),
            flash_until: None,
            embeddings: Vec::new(),
            pending_embeddings: VecDeque::new(),
            logs: VecDeque::new(),
            embed_started_at: None,
            last_embed_log: None,
            embed_done_pending: false,
            last_embed_tick: Instant::now(),
            invert_running: false,
            inversions: Vec::new(),
            show_inversions: false,
            explore_running: false,
            explored_spans: Vec::new(),
            explore_sample_count: 0,
            gpu_ops: VecDeque::new(),
            gpu_ops_per_sec: 0.0,
            negotiate_running: false,
            negotiate_round: 0,
            negotiate_boundary: None,
            negotiate_batch: None,
            show_similarity: false,
            similar_chunks: Vec::new(),
            similarity_scroll: 0,
            similarity_selected: 0,
            focus_left: false,
            expanded_chunk: None,
            source_scroll: 0,
            model_server: None,
            model_request_id: 0,
            show_menu: false,
            menu_selection: 0,
            dim_dissimilar: false,
            search_query: String::new(),
            search_matches: Vec::new(),
            focus_area: FocusArea::ChunkList,
        };
        app.reload()?;
        Ok(app)
    }

    fn reload(&mut self) -> Result<()> {
        let text = if let Some(path) = &self.input_path {
            fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?
        } else if Path::new("hn_frontpage.md").exists() {
            fs::read_to_string("hn_frontpage.md").context("failed to read hn_frontpage.md")?
        } else {
            "Paste text with --input /path/to/file.txt".to_string()
        };

        self.text = clean_text(&text);
        self.processing = false;
        self.processing_pending = false;
        self.auto_embed_pending = false;
        self.process_running = false;
        self.processed_count = 0;
        self.active_index = None;
        self.highlight_scores.clear();
        self.embeddings.clear();
        self.pending_embeddings.clear();
        self.flash_until = None;
        self.logs.clear();
        self.embed_started_at = None;
        self.last_embed_log = None;
        self.embed_done_pending = false;
        self.invert_running = false;
        self.inversions.clear();
        self.show_inversions = false;
        self.explore_running = false;
        self.explored_spans.clear();
        self.explore_sample_count = 0;
        self.gpu_ops.clear();
        self.gpu_ops_per_sec = 0.0;
        self.negotiate_running = false;
        self.negotiate_round = 0;
        self.negotiate_boundary = None;
        self.show_similarity = false;
        self.similar_chunks.clear();
        self.similarity_scroll = 0;
        self.similarity_selected = 0;
        self.focus_left = false;
        self.expanded_chunk = None;
        self.source_scroll = 0;
        let seed = seed_chunks_from_tokens(&self.text, self.window_tokens);
        self.boundary_cutoff = seed.boundary_cutoff;
        self.chunks = seed.chunks;
        self.highlight_scores = vec![0.0; self.chunks.len()];
        self.embeddings = vec![None; self.chunks.len()];
        self.pending_embeddings.clear();
        self.boundary_count = self.chunks.iter().filter(|c| c.is_boundary).count();
        self.score_source = "seed".to_string();
        self.selected = 0.min(self.chunks.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
        self.status = format!(
            "loaded {} chars | seed {} chunks | window {} tokens | auto-embedding...",
            self.text.chars().count(),
            self.chunks.len(),
            self.window_tokens
        );
        self.push_log("seeded chunks, auto-embedding...");
        self.auto_embed_pending = true;
        Ok(())
    }

    fn update_search(&mut self) {
        self.search_matches.clear();
        // Only search if query is at least 2 characters (faster typing)
        if self.search_query.len() < 2 {
            if !self.search_query.is_empty() {
                self.status = "type more to search...".to_string();
            }
            return;
        }
        // Limit search to first 50 matches for performance
        let query_lower = self.search_query.to_lowercase();
        let text_lower = self.text.to_lowercase();
        let mut start = 0;
        let max_matches = 50;
        while let Some(pos) = text_lower[start..].find(&query_lower) {
            let match_start = start + pos;
            let match_end = match_start + self.search_query.len();
            self.search_matches.push((match_start, match_end));
            if self.search_matches.len() >= max_matches {
                break;
            }
            start = match_end;
        }
        if !self.search_matches.is_empty() {
            let suffix = if self.search_matches.len() >= max_matches { "+" } else { "" };
            self.status = format!("{}{} matches", self.search_matches.len(), suffix);
        } else {
            self.status = "no matches".to_string();
        }
    }

    fn adjust_window(&mut self, delta: i32) {
        let next = (self.window_tokens as i32 + delta).max(8) as usize;
        if next != self.window_tokens {
            self.window_tokens = next;
            self.score_config.max_length = next as u32;
            let seed = seed_chunks_from_tokens(&self.text, self.window_tokens);
            self.boundary_cutoff = seed.boundary_cutoff;
            self.chunks = seed.chunks;
            self.highlight_scores = vec![0.0; self.chunks.len()];
            self.embeddings = vec![None; self.chunks.len()];
            self.pending_embeddings.clear();
            self.boundary_count = self.chunks.iter().filter(|c| c.is_boundary).count();
            self.score_source = "seed".to_string();
            self.selected = 0.min(self.chunks.len().saturating_sub(1));
            self.list_state.select(Some(self.selected));
            self.status = format!(
                "window set to {} tokens | seed {} chunks | auto-embedding...",
                self.window_tokens,
                self.chunks.len()
            );
            self.push_log("seeded chunks, auto-embedding...");
            self.auto_embed_pending = true;
        }
    }

    fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.list_state.select(Some(self.selected));
        }
    }

    fn select_next(&mut self) {
        if self.selected + 1 < self.chunks.len() {
            self.selected += 1;
            self.list_state.select(Some(self.selected));
        }
    }

    fn apply_event(&mut self, event: SegmentEvent) {
        match event {
            SegmentEvent::Init { chunks } => {
                self.chunks = chunks
                    .into_iter()
                    .enumerate()
                    .map(|(index, span)| build_chunk_from_span(index, span, &self.text, -1.0))
                    .collect();
                self.highlight_scores = vec![0.0; self.chunks.len()];
                self.embeddings = vec![None; self.chunks.len()];
                self.pending_embeddings.clear();
                self.selected = 0.min(self.chunks.len().saturating_sub(1));
                self.list_state.select(Some(self.selected));
                self.refresh_boundaries();
                self.active_index = None;
                self.flash_until = None;
                self.score_source = "vec2text".to_string();
                self.status = format!("segmenter init: {} chunks", self.chunks.len());
                self.push_log(&format!("segmenter init: {} chunks", self.chunks.len()));
            }
            SegmentEvent::Update {
                index,
                start,
                end,
                score,
            } => {
                let span = SegmentSpan { start, end, score };
                if index < self.chunks.len() {
                    self.chunks[index] = build_chunk_from_span(index, span, &self.text, self.boundary_cutoff);
                    self.selected = index;
                    self.list_state.select(Some(self.selected));
                    self.refresh_boundaries();
                    self.active_index = Some(index);
                    self.flash_until = Some(Instant::now() + Duration::from_millis(300));
                    self.status = format!("segmenter chunk {}/{}", index + 1, self.chunks.len());
                    if index % 5 == 0 {
                        self.push_log(&format!("segmenter update: chunk {}", index + 1));
                    }
                }
            }
            SegmentEvent::Done => {
                self.refresh_boundaries();
                self.active_index = None;
                self.flash_until = None;
                self.status = "segmenter done".to_string();
                self.push_log("segmenter done");
                if self.processing {
                    self.push_log("segmenter updated chunks (re-embed to refresh)");
                }
            }
            SegmentEvent::Error { message } => {
                self.status = format!("segmenter error: {}", message);
                self.push_log(&format!("segmenter error: {}", message));
            }
        }
    }

    fn refresh_boundaries(&mut self) {
        if self.chunks.is_empty() {
            self.boundary_cutoff = 0.0;
            self.boundary_count = 0;
            return;
        }
        let scores: Vec<f64> = self.chunks.iter().map(|c| c.score).collect();
        let cutoff = boundary_cutoff(&scores);
        self.boundary_cutoff = cutoff;
        for chunk in &mut self.chunks {
            chunk.is_boundary = cutoff >= 0.0 && chunk.score <= cutoff;
        }
        self.boundary_count = self.chunks.iter().filter(|c| c.is_boundary).count();
        self.status = format!(
            "loaded {} chars | {} chunks | window {} tokens | boundaries {} | cutoff {:.3} | score {}",
            self.text.chars().count(),
            self.chunks.len(),
            self.window_tokens,
            self.boundary_count,
            self.boundary_cutoff,
            self.score_source
        );
    }

    fn reset_processing_state(&mut self) {
        self.processing = false;
        self.processing_pending = false;
        self.auto_embed_pending = false;
        self.process_running = false;
        self.processed_count = 0;
        self.active_index = None;
        self.flash_until = None;
        self.highlight_scores = vec![0.0; self.chunks.len()];
        self.embeddings = vec![None; self.chunks.len()];
        self.pending_embeddings.clear();
        self.logs.clear();
        self.embed_started_at = None;
        self.last_embed_log = None;
        self.embed_done_pending = false;
    }

    fn record_gpu_op(&mut self) {
        let now = Instant::now();
        self.gpu_ops.push_back(now);
        // Keep only operations from the last 2 seconds
        let cutoff = now - Duration::from_secs(2);
        while let Some(front) = self.gpu_ops.front() {
            if *front < cutoff {
                self.gpu_ops.pop_front();
            } else {
                break;
            }
        }
        // Calculate ops/sec based on the window
        if self.gpu_ops.len() >= 2 {
            if let (Some(first), Some(last)) = (self.gpu_ops.front(), self.gpu_ops.back()) {
                let elapsed = last.duration_since(*first).as_secs_f32();
                if elapsed > 0.0 {
                    self.gpu_ops_per_sec = (self.gpu_ops.len() - 1) as f32 / elapsed;
                }
            }
        } else {
            self.gpu_ops_per_sec = self.gpu_ops.len() as f32;
        }
    }

    fn push_log(&mut self, message: &str) {
        const MAX_LOG_LINES: usize = 6;
        if message.is_empty() {
            return;
        }
        if self.logs.len() >= MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(message.to_string());
    }

    fn update_highlights(&mut self, active_index: usize) {
        self.highlight_scores.fill(0.0);
        let Some(active) = self
            .embeddings
            .get(active_index)
            .and_then(|emb| emb.as_ref())
        else {
            return;
        };
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| emb.as_ref().map(|vec| (idx, cosine_similarity(active, vec))))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = 8usize.min(scored.len());
        if top_k == 0 {
            return;
        }
        for (rank, (idx, _score)) in scored.into_iter().take(top_k).enumerate() {
            let strength = if top_k == 1 {
                1.0
            } else {
                1.0 - (rank as f32) / (top_k as f32 - 1.0)
            };
            if let Some(slot) = self.highlight_scores.get_mut(idx) {
                *slot = strength.max(0.0);
            }
        }
    }

    fn compute_similarity_ranking(&mut self) {
        self.similar_chunks.clear();
        self.similarity_scroll = 0;

        let Some(selected_emb) = self
            .embeddings
            .get(self.selected)
            .and_then(|emb| emb.as_ref())
        else {
            self.push_log("no embedding for selected chunk - embed first");
            return;
        };

        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| {
                emb.as_ref().map(|vec| (idx, cosine_similarity(selected_emb, vec)))
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        self.similar_chunks = scored;
        self.push_log(&format!("computed similarity for {} chunks", self.similar_chunks.len()));
    }

    fn update_processing_status(&mut self) {
        if !self.processing {
            return;
        }
        if self.processed_count > 0 || !self.pending_embeddings.is_empty() {
            return;
        }
        let spinner = ['|', '/', '-', '\\'];
        let idx = (self.last_step.elapsed().as_millis() / 200) as usize % spinner.len();
        let active = self.active_index.map(|i| i + 1).unwrap_or(0);
        self.status = format!(
            "processing {} {}/{} | active {} | window {} tokens",
            spinner[idx],
            self.processed_count,
            self.chunks.len(),
            active,
            self.window_tokens
        );
    }
}

#[derive(Deserialize)]
struct SegmentSpan {
    start: usize,
    end: usize,
    score: f64,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum SegmentEvent {
    Init { chunks: Vec<SegmentSpan> },
    Update { index: usize, start: usize, end: usize, score: f64 },
    Done,
    Error { message: String },
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum ProcessEvent {
    Update { index: usize, start: usize, end: usize, score: f64 },
    Log { message: String },
    Done,
    Error { message: String },
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum EmbedEvent {
    Embedding { index: usize, embedding: Vec<f32> },
    Log { message: String },
    Done,
    Error { message: String },
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum InvertEvent {
    Inversion { index: usize, original: String, inverted: String, score: f64 },
    Log { message: String },
    Done,
    Error { message: String },
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum ExploreEvent {
    Sample { index: usize, start: usize, end: usize, score: f64, preview: String, inverted: String },
    Top { rank: usize, start: usize, end: usize, score: f64, preview: String },
    Log { message: String },
    Done,
    Error { message: String },
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum NegotiateEvent {
    // Phase 1: Claim gaps
    Claim { chunk: usize, gap_start: usize, gap_end: usize, text: String, improvement: f64 },
    // Phase 2: Split
    Split { chunk: usize, at_char: usize, improvement: f64, left_preview: String, right_preview: String },
    // Phase 3: Absorb
    Absorb { chunk: usize, into: usize, reason: String },
    // Phase 4: Give/Birth
    Give { from: usize, to: usize, tokens: usize, text: String },
    Birth { between: Vec<usize>, start: usize, end: usize, says: String, score: f64 },
    // Results
    Final { index: usize, start: usize, end: usize, born: Option<bool> },
    Log { message: String },
    Done,
    Error { message: String },
}

#[derive(Clone)]
struct ScoredSpan {
    start: usize,
    end: usize,
    score: f64,
}

fn clean_text(text: &str) -> String {
    // Replace non-ASCII characters with spaces to avoid byte boundary issues
    text.chars()
        .map(|c| if c.is_ascii() { c } else { ' ' })
        .collect()
}

fn token_offsets(text: &str) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start {
                offsets.push((s, idx));
                start = None;
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(s) = start {
        offsets.push((s, text.len()));
    }
    offsets
}

fn build_token_seeds(text: &str, tokens_per_chunk: usize) -> Vec<ChunkSeed> {
    let offsets = token_offsets(text);
    if offsets.is_empty() {
        return Vec::new();
    }
    let mut seeds = Vec::new();
    let mut idx = 0usize;
    let mut prev_end = 0usize; // Track where last chunk ended for contiguity

    while idx < offsets.len() {
        let end_idx = (idx + tokens_per_chunk).min(offsets.len());
        // Start from where previous chunk ended (include whitespace)
        let start = prev_end;
        // End extends to start of next chunk's first token, or text end
        let end = if end_idx < offsets.len() {
            offsets[end_idx].0 // Start of next token (includes trailing whitespace)
        } else {
            text.len() // Last chunk goes to end of text
        };
        let slice = text.get(start..end).unwrap_or("");
        seeds.push(ChunkSeed {
            start,
            end,
            score: -1.0, // unscored, will be set by vec2text
            text: slice.to_string(),
        });
        prev_end = end;
        idx = end_idx;
    }
    seeds
}



#[derive(Clone)]
struct ScoreConfig {
    python_path: Option<PathBuf>,
    script_path: PathBuf,
    segment_script_path: PathBuf,
    embed_script_path: PathBuf,
    process_script_path: PathBuf,
    invert_script_path: PathBuf,
    explore_script_path: PathBuf,
    negotiate_script_path: PathBuf,
    model: String,
    num_steps: u32,
    max_length: u32,
    embed_batch_size: usize,
}

impl ScoreConfig {
    fn default_with_paths() -> Self {
        let script_path = PathBuf::from("scripts/vec2text_score.py");
        let segment_script_path = PathBuf::from("scripts/vec2text_segment.py");
        let embed_script_path = PathBuf::from("scripts/vec2text_embed.py");
        let process_script_path = PathBuf::from("scripts/vec2text_process.py");
        let invert_script_path = PathBuf::from("scripts/vec2text_invert.py");
        let explore_script_path = PathBuf::from("scripts/vec2text_explore.py");
        let negotiate_script_path = PathBuf::from("scripts/vec2text_negotiate.py");
        let python_path = std::env::var("VECTOR_SMITE_PY")
            .ok()
            .map(PathBuf::from);
        Self {
            python_path,
            script_path,
            segment_script_path,
            embed_script_path,
            process_script_path,
            invert_script_path,
            explore_script_path,
            negotiate_script_path,
            model: "gtr-base".to_string(),
            num_steps: 20,
            max_length: 32,
            embed_batch_size: 16,
        }
    }

    fn is_invert_enabled(&self) -> bool {
        self.invert_script_path.exists()
    }

    fn is_vec2text_enabled(&self) -> bool {
        self.script_path.exists() && self.segment_script_path.exists()
    }

    fn is_embed_enabled(&self) -> bool {
        self.embed_script_path.exists()
    }

    fn is_process_enabled(&self) -> bool {
        self.process_script_path.exists()
    }

    fn is_explore_enabled(&self) -> bool {
        self.explore_script_path.exists()
    }

    fn is_negotiate_enabled(&self) -> bool {
        self.negotiate_script_path.exists()
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    max_length: u32,
    batch_size: usize,
    texts: Vec<String>,
}

#[derive(Serialize)]
struct SegmentRequest {
    text: String,
    max_tokens: usize,
    min_tokens: usize,
    stride: usize,
    shift: usize,
    lengths: Vec<usize>,
    model: String,
    num_steps: u32,
    max_length: u32,
}

#[derive(Serialize)]
struct ProcessChunk {
    start: usize,
    end: usize,
}

#[derive(Serialize)]
struct ProcessRequest {
    text: String,
    chunks: Vec<ProcessChunk>,
    max_tokens: usize,
    min_tokens: usize,
    max_extra_ratio: f64,
    attempts: usize,
    model: String,
    num_steps: u32,
    max_length: u32,
}

#[derive(Serialize)]
struct ExploreRequest {
    text: String,
    num_samples: usize,
    min_tokens: usize,
    max_tokens: usize,
    model: String,
    num_steps: u32,
    max_length: u32,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct NegotiateChunk {
    start: usize,
    end: usize,
}

#[derive(Serialize)]
struct NegotiateRequest {
    text: String,
    chunks: Vec<NegotiateChunk>,
    focus_chunk: usize,  // Which chunk to negotiate from
    max_shift: usize,
    min_tokens: usize,
    model: String,
    max_length: u32,
}

fn boundary_cutoff(scores: &[f64]) -> f64 {
    // Only consider scored chunks (score >= 0)
    let valid: Vec<f64> = scores.iter().copied().filter(|&s| s >= 0.0).collect();
    if valid.is_empty() {
        return -1.0; // No valid scores yet
    }
    let mut sorted = valid;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = 0.2;
    let idx = ((sorted.len() - 1) as f64 * quantile).floor() as usize;
    sorted[idx]
}

fn seed_chunks_from_tokens(text: &str, window_tokens: usize) -> SeedAnalysis {
    let seeds = build_token_seeds(text, window_tokens);
    let cutoff = if seeds.len() >= 3 {
        boundary_cutoff(&seeds.iter().map(|s| s.score).collect::<Vec<_>>())
    } else {
        -1.0
    };
    let chunks = seeds
        .into_iter()
        .enumerate()
        .map(|(index, seed)| build_chunk(index, seed, cutoff))
        .collect();
    SeedAnalysis {
        chunks,
        boundary_cutoff: cutoff,
    }
}

fn build_chunk(index: usize, seed: ChunkSeed, cutoff: f64) -> ChunkInfo {
    let preview = snip(&seed.text, 80);
    let is_boundary = cutoff >= 0.0 && seed.score <= cutoff;
    ChunkInfo {
        index,
        start: seed.start,
        end: seed.end,
        score: seed.score,
        is_boundary,
        preview,
        text: seed.text,
    }
}

fn build_chunk_from_span(index: usize, span: SegmentSpan, text: &str, cutoff: f64) -> ChunkInfo {
    let start = span.start.min(text.len());
    let end = span.end.min(text.len());
    let slice = if end > start { &text[start..end] } else { "" };
    let preview = snip(slice, 80);
    let is_boundary = cutoff >= 0.0 && span.score <= cutoff;
    ChunkInfo {
        index,
        start,
        end,
        score: span.score,
        is_boundary,
        preview,
        text: slice.to_string(),
    }
}

fn snip(text: &str, limit: usize) -> String {
    let mut cleaned = String::new();
    let mut last_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_space {
                cleaned.push(' ');
                last_space = true;
            }
        } else {
            cleaned.push(ch);
            last_space = false;
        }
    }
    let cleaned = cleaned.trim();
    let mut out = String::new();
    for (i, ch) in cleaned.chars().enumerate() {
        if i >= limit {
            out.push_str("...");
            break;
        }
        out.push(ch);
    }
    out
}

fn score_style(chunk: &ChunkInfo) -> Style {
    if chunk.score < 0.0 {
        // Unscored chunk - neutral gray
        Style::default().fg(Color::Gray)
    } else if chunk.is_boundary {
        Style::default()
            .fg(Color::Red)
            .add_modifier(Modifier::BOLD)
    } else if chunk.score < 0.2 {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn highlight_color(strength: f32) -> Color {
    let clamped = strength.clamp(0.0, 1.0);
    let start = (255.0, 140.0, 0.0);
    let end = (255.0, 105.0, 180.0);
    let r = (start.0 + (end.0 - start.0) * clamped).round() as u8;
    let g = (start.1 + (end.1 - start.1) * clamped).round() as u8;
    let b = (start.2 + (end.2 - start.2) * clamped).round() as u8;
    Color::Rgb(r, g, b)
}

fn spawn_model_server(config: &ScoreConfig) -> Result<ModelServer> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };

    // Find model_server.py in scripts directory
    let server_script = config.embed_script_path
        .parent()
        .map(|p| p.join("model_server.py"))
        .unwrap_or_else(|| PathBuf::from("scripts/model_server.py"));

    let mut child = Command::new(&python)
        .arg(&server_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start model server")?;

    let mut stdin = child.stdin.take().context("failed to get model server stdin")?;
    let stdout = child.stdout.take().context("failed to get model server stdout")?;

    // Send initial config
    let config_json = serde_json::json!({
        "model": config.model,
    });
    writeln!(stdin, "{}", config_json).context("failed to send config to model server")?;
    stdin.flush()?;

    // Create channel for responses
    let (tx, rx) = mpsc::channel();

    // Background thread to read responses
    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines().flatten() {
            if let Ok(response) = serde_json::from_str::<ModelResponse>(&line) {
                if tx.send(response).is_err() {
                    break;
                }
            }
        }
    });

    Ok(ModelServer {
        stdin,
        response_rx: rx,
        child,
    })
}

impl ModelServer {
    fn send_command(&mut self, cmd: &ModelCommand) -> Result<()> {
        let json = serde_json::to_string(cmd)?;
        writeln!(self.stdin, "{}", json)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn try_recv(&self) -> Option<ModelResponse> {
        self.response_rx.try_recv().ok()
    }
}

/// Sanitize text for embedding - replace non-ASCII with spaces
fn sanitize_for_embed(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_ascii() { c } else { ' ' })
        .collect()
}

fn spawn_embedder(texts: Vec<String>, config: &ScoreConfig) -> Result<Receiver<EmbedEvent>> {
    let texts: Vec<String> = texts.iter().map(|t| sanitize_for_embed(t)).collect();
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let embed_script = config.embed_script_path.clone();
    let model = config.model.clone();
    let max_length = config.max_length;
    let request = EmbedRequest {
        model,
        max_length,
        batch_size: config.embed_batch_size,
        texts,
    };
    let payload = serde_json::to_vec(&request)?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut child = match Command::new(python)
            .arg(&embed_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.send(EmbedEvent::Error {
                    message: format!("failed to start embedder: {}", err),
                });
                return;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(EmbedEvent::Error {
                    message: "failed to write embedder payload".to_string(),
                });
                return;
            }
        }

        // Capture stderr in a separate thread
        let stderr_handle = child.stderr.take();
        let stderr_thread = std::thread::spawn(move || {
            let mut stderr_output = String::new();
            if let Some(mut err) = stderr_handle {
                let _ = err.read_to_string(&mut stderr_output);
            }
            stderr_output
        });

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<EmbedEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        }

        let stderr_output = stderr_thread.join().unwrap_or_default();
        let status = child.wait();
        if let Ok(exit) = status {
            if !exit.success() {
                // Log each line of stderr separately for visibility
                for line in stderr_output.lines().take(20) {
                    let _ = tx.send(EmbedEvent::Log {
                        message: format!("stderr: {}", line),
                    });
                }
                let _ = tx.send(EmbedEvent::Error {
                    message: format!("embedder exit {}", exit),
                });
            } else {
                let _ = tx.send(EmbedEvent::Done);
            }
        }
    });

    Ok(rx)
}

#[derive(Serialize)]
struct InvertRequest {
    model: String,
    num_steps: u32,
    max_length: u32,
    texts: Vec<String>,
}

fn spawn_inverter(texts: Vec<String>, config: &ScoreConfig) -> Result<Receiver<InvertEvent>> {
    let texts: Vec<String> = texts.iter().map(|t| sanitize_for_embed(t)).collect();
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let invert_script = config.invert_script_path.clone();
    let request = InvertRequest {
        model: config.model.clone(),
        num_steps: config.num_steps,
        max_length: config.max_length,
        texts,
    };
    let payload = serde_json::to_vec(&request)?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut child = match Command::new(python)
            .arg(&invert_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.send(InvertEvent::Error {
                    message: format!("failed to start inverter: {}", err),
                });
                return;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(InvertEvent::Error {
                    message: "failed to write inverter payload".to_string(),
                });
                return;
            }
        }

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<InvertEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        }

        let status = child.wait();
        if let Ok(exit) = status {
            if !exit.success() {
                let mut stderr = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                let _ = tx.send(InvertEvent::Error {
                    message: format!("inverter exit {} {}", exit, stderr),
                });
            } else {
                let _ = tx.send(InvertEvent::Done);
            }
        }
    });

    Ok(rx)
}

fn spawn_segmenter(text: String, config: &ScoreConfig) -> Result<Receiver<SegmentEvent>> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let segment_script = config.segment_script_path.clone();
    let model = config.model.clone();
    let max_length = config.max_length;
    let num_steps = config.num_steps;
    let request = SegmentRequest {
        text: sanitize_for_embed(&text),
        max_tokens: max_length as usize,
        min_tokens: 12,
        stride: max_length as usize,
        shift: 4,
        lengths: vec![
            max_length as usize,
            max_length.saturating_sub(4) as usize,
            max_length.saturating_sub(8) as usize,
        ]
        .into_iter()
        .filter(|len| *len >= 12)
        .collect(),
        model,
        num_steps,
        max_length,
    };
    let payload = serde_json::to_vec(&request)?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut child = match Command::new(python)
            .arg(&segment_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.send(SegmentEvent::Error {
                    message: format!("failed to start segmenter: {}", err),
                });
                return;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(SegmentEvent::Error {
                    message: "failed to write segmenter payload".to_string(),
                });
                return;
            }
        }

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<SegmentEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        }

        let status = child.wait();
        if let Ok(exit) = status {
            if !exit.success() {
                let mut stderr = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                let _ = tx.send(SegmentEvent::Error {
                    message: format!("segmenter exit {} {}", exit, stderr),
                });
            } else {
                let _ = tx.send(SegmentEvent::Done);
            }
        }
    });

    Ok(rx)
}

fn spawn_processer(text: String, chunks: &[ChunkInfo], config: &ScoreConfig) -> Result<Receiver<ProcessEvent>> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let process_script = config.process_script_path.clone();
    let model = config.model.clone();
    let max_length = config.max_length;
    let request = ProcessRequest {
        text: sanitize_for_embed(&text),
        chunks: chunks
            .iter()
            .map(|chunk| ProcessChunk {
                start: chunk.start,
                end: chunk.end,
            })
            .collect(),
        max_tokens: max_length as usize,
        min_tokens: 8,
        max_extra_ratio: 0.75,
        attempts: 6,
        model,
        num_steps: config.num_steps,
        max_length,
    };
    let payload = serde_json::to_vec(&request)?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut child = match Command::new(python)
            .arg(&process_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.send(ProcessEvent::Error {
                    message: format!("failed to start processer: {}", err),
                });
                return;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(ProcessEvent::Error {
                    message: "failed to write process payload".to_string(),
                });
                return;
            }
        }

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<ProcessEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        }

        let status = child.wait();
        if let Ok(exit) = status {
            if !exit.success() {
                let mut stderr = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                let _ = tx.send(ProcessEvent::Error {
                    message: format!("process exit {} {}", exit, stderr),
                });
            } else {
                let _ = tx.send(ProcessEvent::Done);
            }
        }
    });

    Ok(rx)
}

fn spawn_explorer(text: String, config: &ScoreConfig, num_samples: usize) -> Result<Receiver<ExploreEvent>> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let explore_script = config.explore_script_path.clone();
    let request = ExploreRequest {
        text: sanitize_for_embed(&text),
        num_samples,
        min_tokens: 8,
        max_tokens: config.max_length as usize,
        model: config.model.clone(),
        num_steps: config.num_steps,
        max_length: config.max_length,
        seed: None,
    };
    let payload = serde_json::to_vec(&request)?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut child = match Command::new(python)
            .arg(&explore_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.send(ExploreEvent::Error {
                    message: format!("failed to start explorer: {}", err),
                });
                return;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(ExploreEvent::Error {
                    message: "failed to write explorer payload".to_string(),
                });
                return;
            }
        }

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<ExploreEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        }

        let status = child.wait();
        if let Ok(exit) = status {
            if !exit.success() {
                let mut stderr = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut stderr);
                }
                let _ = tx.send(ExploreEvent::Error {
                    message: format!("explorer exit {} {}", exit, stderr),
                });
            } else {
                let _ = tx.send(ExploreEvent::Done);
            }
        }
    });

    Ok(rx)
}

fn spawn_negotiator(
    text: String,
    chunks: &[ChunkInfo],
    focus_chunk: usize,
    config: &ScoreConfig,
) -> Result<(Receiver<NegotiateEvent>, Child)> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let negotiate_script = config.negotiate_script_path.clone();
    let request = NegotiateRequest {
        text: sanitize_for_embed(&text),
        chunks: chunks
            .iter()
            .map(|c| NegotiateChunk {
                start: c.start,
                end: c.end,
            })
            .collect(),
        focus_chunk,
        max_shift: 8,
        min_tokens: NEGOTIATE_MIN_TOKENS,
        model: config.model.clone(),
        max_length: config.max_length,
    };
    let payload = serde_json::to_vec(&request)?;

    let mut child = Command::new(python)
        .arg(&negotiate_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start negotiator")?;

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        if let Err(err) = stdin.write_all(&payload) {
            stop_child_process(&mut child);
            return Err(err).context("failed to write negotiator payload");
        }
    }

    let (tx, rx) = mpsc::channel();
    if let Some(stdout) = child.stdout.take() {
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<NegotiateEvent>(&line) {
                    let _ = tx.send(event);
                }
            }
        });
    }

    Ok((rx, child))
}

fn stop_child_process(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn stop_negotiate_process(proc: &mut Option<NegotiateProcess>) {
    if let Some(mut running) = proc.take() {
        stop_child_process(&mut running.child);
    }
}

fn chunk_ready_for_negotiation(chunk: &ChunkInfo) -> bool {
    chunk.text.split_whitespace().count() >= NEGOTIATE_MIN_WORDS
}

fn start_negotiation(
    app: &mut App,
    negotiate_proc: &mut Option<NegotiateProcess>,
    focus_chunk: usize,
) -> Result<()> {
    if app.chunks.is_empty() {
        return Ok(());
    }
    let focus_chunk = focus_chunk.min(app.chunks.len().saturating_sub(1));
    app.selected = focus_chunk;
    app.list_state.select(Some(app.selected));
    app.negotiate_running = true;
    app.negotiate_round = 0;
    app.negotiate_boundary = None;
    match spawn_negotiator(app.text.clone(), &app.chunks, focus_chunk, &app.score_config) {
        Ok((rx, child)) => {
            *negotiate_proc = Some(NegotiateProcess { rx, child });
            Ok(())
        }
        Err(err) => {
            app.negotiate_running = false;
            Err(err)
        }
    }
}

fn start_negotiate_all(app: &mut App, negotiate_proc: &mut Option<NegotiateProcess>) {
    if app.negotiate_running || app.negotiate_batch.is_some() {
        app.push_log("negotiation already running");
        app.status = "negotiation already running".to_string();
        return;
    }
    if app.chunks.len() < 2 {
        app.push_log("need at least 2 chunks to negotiate");
        app.status = "need at least 2 chunks".to_string();
        return;
    }
    if !app.score_config.is_negotiate_enabled() {
        app.push_log("negotiate script not found");
        app.status = "negotiate script not found".to_string();
        return;
    }

    let mut queue = VecDeque::new();
    let mut skipped = 0usize;
    for (idx, chunk) in app.chunks.iter().enumerate() {
        if chunk_ready_for_negotiation(chunk) {
            queue.push_back(idx);
        } else {
            skipped += 1;
        }
    }

    if queue.is_empty() {
        app.push_log("no chunks long enough to negotiate");
        app.status = "no chunks long enough to negotiate".to_string();
        return;
    }

    let total = queue.len();
    app.negotiate_batch = Some(NegotiateBatch { queue, total, skipped });
    app.push_log(&format!("negotiating all: {} chunks (skipped {})", total, skipped));
    advance_negotiate_batch(app, negotiate_proc);
}

fn advance_negotiate_batch(app: &mut App, negotiate_proc: &mut Option<NegotiateProcess>) {
    if app.negotiate_running {
        return;
    }
    let mut batch = match app.negotiate_batch.take() {
        Some(batch) => batch,
        None => return,
    };

    let Some(next_index) = batch.queue.pop_front() else {
        app.status = format!(
            "negotiation batch done | {} chunks (skipped {})",
            batch.total,
            batch.skipped
        );
        app.push_log(&format!("negotiation batch done (skipped {})", batch.skipped));
        return;
    };

    let total = batch.total;
    let skipped = batch.skipped;
    let current = total.saturating_sub(batch.queue.len());

    if let Err(err) = start_negotiation(app, negotiate_proc, next_index) {
        app.status = format!("negotiate failed: {}", err);
        app.push_log("negotiate failed to start");
        return;
    }

    app.negotiate_batch = Some(batch);
    app.status = format!("negotiating all: {}/{} (skipped {})", current, total, skipped);
    app.push_log(&format!(
        "negotiating chunk {}/{} (#{})",
        current,
        total,
        next_index + 1
    ));
}

fn chunk_palette(index: usize) -> Color {
    let palette = [
        Color::LightBlue,
        Color::LightGreen,
        Color::LightCyan,
        Color::LightMagenta,
        Color::LightYellow,
        Color::Cyan,
        Color::Green,
    ];
    palette[index % palette.len()]
}

fn chunk_style(app: &App, chunk: &ChunkInfo, selected: bool) -> Style {
    let mut style = Style::default()
        .bg(chunk_palette(chunk.index))
        .fg(Color::Black);
    let highlight = app
        .highlight_scores
        .get(chunk.index)
        .copied()
        .unwrap_or(0.0);
    let now = Instant::now();
    let is_active = app.active_index == Some(chunk.index);
    let flashing = app
        .flash_until
        .map(|until| until > now)
        .unwrap_or(false)
        && is_active;
    if flashing {
        style = Style::default()
            .bg(Color::Black)
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    } else if highlight > 0.0 {
        style = Style::default()
            .bg(highlight_color(highlight))
            .fg(Color::Black);
    } else if chunk.is_boundary {
        style = Style::default()
            .bg(Color::Red)
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
    }
    if is_active {
        style = style.add_modifier(Modifier::BOLD);
    }
    if selected {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

fn list_style(app: &App, chunk: &ChunkInfo) -> Style {
    // Dim dissimilar chunks when showing similarity
    if app.dim_dissimilar {
        // Find this chunk's similarity score
        let sim = app.similar_chunks
            .iter()
            .find(|(idx, _)| *idx == chunk.index)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        if chunk.index == app.selected {
            // Selected chunk (the one we're comparing to)
            return Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        } else if sim > 0.8 {
            // Very similar - bright
            return Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
        } else if sim > 0.6 {
            // Somewhat similar
            return Style::default().fg(Color::Yellow);
        } else if sim > 0.4 {
            // Less similar - dim
            return Style::default().fg(Color::DarkGray);
        } else {
            // Not similar - very dim
            return Style::default().fg(Color::Rgb(60, 60, 60));
        }
    }

    chunk_style(app, chunk, app.selected == chunk.index)
}

/// Wrap text by inserting newlines at word boundaries
fn wrap_text(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return text.to_string();
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if word_len > max_width {
            if current_len > 0 {
                lines.push(std::mem::take(&mut current));
                current_len = 0;
            }

            let mut chunk = String::new();
            let mut chunk_len = 0usize;
            for ch in word.chars() {
                chunk.push(ch);
                chunk_len += 1;
                if chunk_len == max_width {
                    lines.push(std::mem::take(&mut chunk));
                    chunk_len = 0;
                }
            }

            if !chunk.is_empty() {
                current = chunk;
                current_len = chunk_len;
            }
        } else if current_len == 0 {
            current.push_str(word);
            current_len = word_len;
        } else if current_len + 1 + word_len <= max_width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
            current_len = word_len;
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }

    lines.join("\n")
}

fn flush_span<'a>(
    spans: &mut Vec<Span<'a>>,
    text: &mut String,
    style: &mut Option<Style>,
) {
    if !text.is_empty() {
        let span_style = style.unwrap_or_default();
        spans.push(Span::styled(std::mem::take(text), span_style));
    }
}

fn push_wrapped_line<'a>(
    wrapped: &mut Vec<Line<'a>>,
    spans: &mut Vec<Span<'a>>,
    text: &mut String,
    style: &mut Option<Style>,
) {
    flush_span(spans, text, style);
    if spans.is_empty() {
        wrapped.push(Line::from(""));
    } else {
        wrapped.push(Line::from(std::mem::take(spans)));
    }
}

fn wrap_lines_with_marker<'a>(
    lines: Vec<Line<'a>>,
    max_width: usize,
    target_marker: Option<usize>,
) -> (Vec<Line<'a>>, Option<usize>) {
    if max_width == 0 {
        return (lines, None);
    }

    let mut wrapped: Vec<Line<'a>> = Vec::new();
    let mut marker_count = 0usize;
    let mut selected_line = None;

    for line in lines {
        let mut current_spans: Vec<Span<'a>> = Vec::new();
        let mut current_span_text = String::new();
        let mut current_span_style: Option<Style> = None;
        let mut current_width = 0usize;
        let mut line_had_content = false;
        let mut pushed_any = false;

        for span in line.spans {
            let span_style = span.style;
            for ch in span.content.chars() {
                line_had_content = true;
                let mut ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
                if ch_width == 0 {
                    ch_width = 1;
                }
                if current_width + ch_width > max_width && current_width > 0 {
                    push_wrapped_line(
                        &mut wrapped,
                        &mut current_spans,
                        &mut current_span_text,
                        &mut current_span_style,
                    );
                    pushed_any = true;
                    current_width = 0;
                }
                if current_width == 0 && ch.is_whitespace() {
                    continue;
                }
                if current_span_style != Some(span_style) {
                    flush_span(&mut current_spans, &mut current_span_text, &mut current_span_style);
                    current_span_style = Some(span_style);
                }
                if ch == '«' {
                    if let Some(target) = target_marker {
                        if marker_count == target && selected_line.is_none() {
                            selected_line = Some(wrapped.len());
                        }
                    }
                    marker_count += 1;
                }
                current_span_text.push(ch);
                current_width += ch_width;
            }
        }

        if line_had_content {
            if !current_span_text.is_empty() || !current_spans.is_empty() {
                push_wrapped_line(
                    &mut wrapped,
                    &mut current_spans,
                    &mut current_span_text,
                    &mut current_span_style,
                );
            } else if !pushed_any {
                wrapped.push(Line::from(""));
            }
        } else {
            wrapped.push(Line::from(""));
        }
    }

    (wrapped, selected_line)
}

fn push_span<'a>(
    spans: &mut Vec<Span<'a>>,
    text: String,
    style: Style,
) {
    spans.push(Span::styled(text, style));
}

fn append_text<'a>(
    lines: &mut Vec<Line<'a>>,
    spans: &mut Vec<Span<'a>>,
    text: &str,
    style: Style,
) {
    let mut buf = String::new();
    let mut last_was_cr = false;
    for ch in text.chars() {
        if ch == '\r' {
            if !buf.is_empty() {
                push_span(spans, buf.clone(), style);
                buf.clear();
            }
            if !spans.is_empty() {
                lines.push(Line::from(std::mem::take(spans)));
            } else {
                lines.push(Line::from(""));
            }
            last_was_cr = true;
            continue;
        }
        if ch == '\n' {
            if last_was_cr {
                last_was_cr = false;
                continue;
            }
            if !buf.is_empty() {
                push_span(spans, buf.clone(), style);
                buf.clear();
            }
            if !spans.is_empty() {
                lines.push(Line::from(std::mem::take(spans)));
            } else {
                lines.push(Line::from(""));
            }
            continue;
        }
        last_was_cr = false;
        buf.push(ch);
    }
    if !buf.is_empty() {
        push_span(spans, buf, style);
    }
}

/// Append text with search highlighting
fn append_text_with_highlight<'a>(
    lines: &mut Vec<Line<'a>>,
    spans: &mut Vec<Span<'a>>,
    text: &str,
    text_start: usize,
    base_style: Style,
    search_matches: &[(usize, usize)],
) {
    let highlight_style = Style::default().fg(Color::Black).bg(Color::Yellow);
    let text_end = text_start + text.len();

    // Find overlapping matches
    let relevant_matches: Vec<(usize, usize)> = search_matches
        .iter()
        .filter(|(ms, me)| *ms < text_end && *me > text_start)
        .cloned()
        .collect();

    if relevant_matches.is_empty() {
        // No matches, use normal append
        append_text(lines, spans, text, base_style);
        return;
    }

    // Process text character by character with highlighting
    let chars: Vec<char> = text.chars().collect();
    let mut char_offsets: Vec<usize> = Vec::with_capacity(chars.len() + 1);
    let mut byte_offset = 0;
    for ch in &chars {
        char_offsets.push(byte_offset);
        byte_offset += ch.len_utf8();
    }
    char_offsets.push(byte_offset);

    let mut buf = String::new();
    let mut current_highlight = false;
    let mut last_was_cr = false;

    for (i, &ch) in chars.iter().enumerate() {
        let global_pos = text_start + char_offsets[i];
        let in_match = relevant_matches.iter().any(|(ms, me)| global_pos >= *ms && global_pos < *me);

        if ch == '\r' || ch == '\n' {
            if !buf.is_empty() {
                let style = if current_highlight { highlight_style } else { base_style };
                push_span(spans, buf.clone(), style);
                buf.clear();
            }
            if !spans.is_empty() {
                lines.push(Line::from(std::mem::take(spans)));
            } else {
                lines.push(Line::from(""));
            }
            if ch == '\r' {
                last_was_cr = true;
            } else if last_was_cr {
                last_was_cr = false;
            }
            current_highlight = false;
            continue;
        }
        last_was_cr = false;

        if in_match != current_highlight {
            // Style changed, flush buffer
            if !buf.is_empty() {
                let style = if current_highlight { highlight_style } else { base_style };
                push_span(spans, buf.clone(), style);
                buf.clear();
            }
            current_highlight = in_match;
        }
        buf.push(ch);
    }

    if !buf.is_empty() {
        let style = if current_highlight { highlight_style } else { base_style };
        push_span(spans, buf, style);
    }
}

fn build_overview_lines(app: &App) -> Vec<Line> {
    if app.text.is_empty() {
        return vec![Line::from("no text loaded")];
    }

    let mut lines: Vec<Line> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut cursor = 0usize;

    let mut chunks = app.chunks.clone();
    chunks.sort_by_key(|chunk| chunk.start);

    let search_matches = &app.search_matches;

    for chunk in chunks {
        let start = chunk.start.min(app.text.len());
        let end = chunk.end.min(app.text.len());
        if start > cursor {
            if let Some(gap) = app.text.get(cursor..start) {
                // Gap text with search highlighting
                append_text_with_highlight(&mut lines, &mut spans, gap, cursor, Style::default().fg(Color::Gray), search_matches);
            }
        }
        if end > start {
            let style = chunk_style(app, &chunk, app.selected == chunk.index);

            // Show inverted text if available and toggled on
            if app.show_inversions {
                if let Some(Some(inversion)) = app.inversions.get(chunk.index) {
                    let inv_style = Style::default().fg(Color::Magenta).bg(Color::Black);
                    let wrapped = wrap_text(&inversion.inverted, 50);
                    append_text(&mut lines, &mut spans, &wrapped, inv_style);
                } else if let Some(segment) = app.text.get(start..end) {
                    let dim_style = Style::default().fg(Color::DarkGray);
                    append_text_with_highlight(&mut lines, &mut spans, segment, start, dim_style, search_matches);
                }
            } else if let Some(segment) = app.text.get(start..end) {
                append_text_with_highlight(&mut lines, &mut spans, segment, start, style, search_matches);
            }
        }
        cursor = end;
    }

    if cursor < app.text.len() {
        if let Some(rest) = app.text.get(cursor..) {
            // Trailing text with search highlighting
            append_text_with_highlight(&mut lines, &mut spans, rest, cursor, Style::default().fg(Color::Gray), search_matches);
        }
    }

    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }

    lines
}

fn setup_terminal() -> Result<Terminal<ratatui::backend::CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    Terminal::new(backend).map_err(Into::into)
}

fn restore_terminal(mut terminal: Terminal<ratatui::backend::CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut score_config = ScoreConfig::default_with_paths();
    while let Some(arg) = args.next() {
        if arg == "--input" || arg == "-i" {
            if let Some(path) = args.next() {
                input_path = Some(PathBuf::from(path));
            }
        } else if arg == "--py" {
            if let Some(path) = args.next() {
                score_config.python_path = Some(PathBuf::from(path));
            }
        }
    }

    let mut app = App::new(input_path, score_config.max_length as usize, score_config)?;
    let mut segment_rx: Option<Receiver<SegmentEvent>> = None;
    let mut embed_rx: Option<Receiver<EmbedEvent>> = None;
    let mut process_rx: Option<Receiver<ProcessEvent>> = None;
    let mut invert_rx: Option<Receiver<InvertEvent>> = None;
    let mut explore_rx: Option<Receiver<ExploreEvent>> = None;
    let mut negotiate_proc: Option<NegotiateProcess> = None;
    let mut terminal = setup_terminal()?;
    let tick_rate = Duration::from_millis(200);
    let mut last_tick = Instant::now();

    loop {
        if let Some(rx) = &segment_rx {
            while let Ok(event) = rx.try_recv() {
                app.apply_event(event);
            }
        }
        // Handle invert events - auto-recover if flag stuck
        if app.invert_running && invert_rx.is_none() {
            app.invert_running = false;
            app.push_log("invert state recovered (process ended unexpectedly)");
        }
        if let Some(rx) = invert_rx.as_ref() {
            let mut invert_done = false;
            let mut invert_error: Option<String> = None;
            while let Ok(event) = rx.try_recv() {
                match event {
                    InvertEvent::Inversion { index, original, inverted, score } => {
                        if index < app.inversions.len() {
                            app.inversions[index] = Some(Inversion {
                                index,
                                original,
                                inverted: inverted.clone(),
                                score,
                            });
                            app.active_index = Some(index);
                            app.flash_until = Some(Instant::now() + Duration::from_millis(300));
                            app.record_gpu_op();
                            app.push_log(&format!("inverted {}: {} -> {}", index + 1,
                                &app.chunks.get(index).map(|c| c.preview.chars().take(20).collect::<String>()).unwrap_or_default(),
                                inverted.chars().take(30).collect::<String>()));
                        }
                    }
                    InvertEvent::Log { message } => {
                        app.push_log(&message);
                    }
                    InvertEvent::Done => {
                        invert_done = true;
                    }
                    InvertEvent::Error { message } => {
                        invert_error = Some(message);
                    }
                }
            }
            if invert_done {
                app.invert_running = false;
                app.show_inversions = true;
                app.status = "inversion done - press 'v' to toggle view".to_string();
                app.push_log("inversion complete");
                invert_rx = None;
            } else if let Some(message) = invert_error {
                app.invert_running = false;
                app.status = format!("invert error: {}", message);
                app.push_log("invert error");
                invert_rx = None;
            }
        }
        // Handle explore events
        if let Some(rx) = explore_rx.as_ref() {
            let mut explore_done = false;
            let mut explore_error: Option<String> = None;
            while let Ok(event) = rx.try_recv() {
                match event {
                    ExploreEvent::Sample { index, start, end, score, preview, inverted: _ } => {
                        app.explored_spans.push(ScoredSpan { start, end, score });
                        app.explore_sample_count = index + 1;
                        app.active_index = Some(index % app.chunks.len().max(1));
                        app.flash_until = Some(Instant::now() + Duration::from_millis(150));
                        app.record_gpu_op();
                        if (index + 1) % 10 == 0 {
                            app.push_log(&format!("sampled {}: score={:.3} \"{}\"",
                                index + 1, score, preview.chars().take(30).collect::<String>()));
                        }
                        app.status = format!("exploring {}/{} samples | score {:.3}",
                            app.explore_sample_count, 100, score);
                    }
                    ExploreEvent::Top { rank, start, end, score, preview } => {
                        app.push_log(&format!("top {}: [{}-{}] score={:.3} \"{}\"",
                            rank, start, end, score, preview.chars().take(40).collect::<String>()));
                    }
                    ExploreEvent::Log { message } => {
                        app.push_log(&message);
                    }
                    ExploreEvent::Done => {
                        explore_done = true;
                    }
                    ExploreEvent::Error { message } => {
                        explore_error = Some(message);
                    }
                }
            }
            if explore_done {
                app.explore_running = false;
                app.status = format!("exploration done | {} samples | {} best spans",
                    app.explore_sample_count,
                    app.explored_spans.iter().filter(|s| s.score > 0.9).count());
                app.push_log(&format!("exploration complete: {} samples", app.explore_sample_count));
                explore_rx = None;
            } else if let Some(message) = explore_error {
                app.explore_running = false;
                app.status = format!("explore error: {}", message);
                app.push_log("explore error");
                explore_rx = None;
            }
        }
        // Handle negotiate events
        let mut negotiate_done = false;
        let mut negotiate_error: Option<String> = None;
        if let Some(proc) = negotiate_proc.as_mut() {
            while let Ok(event) = proc.rx.try_recv() {
                match event {
                    NegotiateEvent::Claim { chunk, gap_start, gap_end, text, improvement } => {
                        app.record_gpu_op();
                        app.active_index = Some(chunk);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(200));
                        let sign = if improvement >= 0.0 { "+" } else { "" };
                        app.push_log(&format!("CLAIM: #{} takes gap [{}-{}] \"{}\" ({}{:.3})",
                            chunk + 1, gap_start, gap_end, text.chars().take(25).collect::<String>(), sign, improvement));
                        app.status = format!("phase 1: claiming gaps...");
                    }
                    NegotiateEvent::Split { chunk, at_char, improvement, left_preview, right_preview } => {
                        app.record_gpu_op();
                        app.active_index = Some(chunk);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(400));
                        app.push_log(&format!("SPLIT: #{} at {} (+{:.3}) \"{}\" | \"{}\"",
                            chunk + 1, at_char, improvement,
                            left_preview.chars().take(20).collect::<String>(),
                            right_preview.chars().take(20).collect::<String>()));
                        app.status = format!("phase 2: chunk {} split into two", chunk + 1);
                    }
                    NegotiateEvent::Absorb { chunk, into, reason } => {
                        app.record_gpu_op();
                        app.active_index = Some(into);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(300));
                        app.push_log(&format!("ABSORB: #{} into #{} ({})", chunk + 1, into + 1, reason));
                        app.status = format!("phase 3: chunk {} absorbed", chunk + 1);
                    }
                    NegotiateEvent::Give { from, to, tokens, text } => {
                        app.record_gpu_op();
                        app.active_index = Some(from);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(300));
                        app.push_log(&format!("GIVE: #{} gives {} tokens to #{}: \"{}\"",
                            from + 1, tokens, to + 1, text.chars().take(25).collect::<String>()));
                        app.status = format!("phase 4: #{} gives to #{}", from + 1, to + 1);
                    }
                    NegotiateEvent::Birth { between, start, end, says, score } => {
                        app.record_gpu_op();
                        let parents = if between.len() >= 2 {
                            format!("#{} and #{}", between[0] + 1, between[1] + 1)
                        } else {
                            "unknown".to_string()
                        };
                        app.push_log(&format!("BIRTH: new chunk between {} [{}-{}] \"{:.25}\" ({:.3})",
                            parents, start, end, says, score));
                        app.status = format!("phase 4: new chunk born between {}", parents);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(400));
                    }
                    NegotiateEvent::Final { index, start, end, born } => {
                        // For born chunks, we need to insert; for existing, we update
                        if born.unwrap_or(false) {
                            // This is a newly born chunk - insert it
                            let span = SegmentSpan { start, end, score: 0.5 }; // Default score for new chunks
                            let new_chunk = build_chunk_from_span(index, span, &app.text, app.boundary_cutoff);
                            if index <= app.chunks.len() {
                                app.chunks.insert(index, new_chunk);
                                // Re-index all chunks after insertion
                                for (i, chunk) in app.chunks.iter_mut().enumerate() {
                                    chunk.index = i;
                                }
                                app.embeddings.insert(index, None);
                                app.highlight_scores.insert(index, 0.0);
                                if app.inversions.len() > 0 {
                                    app.inversions.insert(index, None);
                                }
                            }
                        } else if index < app.chunks.len() {
                            let span = SegmentSpan { start, end, score: app.chunks[index].score };
                            app.chunks[index] = build_chunk_from_span(index, span, &app.text, app.boundary_cutoff);
                        }
                    }
                    NegotiateEvent::Log { message } => {
                        app.push_log(&message);
                    }
                    NegotiateEvent::Done => {
                        negotiate_done = true;
                    }
                    NegotiateEvent::Error { message } => {
                        negotiate_error = Some(message);
                    }
                }
            }
        }
        if negotiate_done {
            app.negotiate_running = false;
            app.negotiate_boundary = None;
            app.refresh_boundaries();
            app.status = format!("negotiation done | {} rounds", app.negotiate_round);
            app.push_log("negotiation complete");
            stop_negotiate_process(&mut negotiate_proc);
            advance_negotiate_batch(&mut app, &mut negotiate_proc);
        } else if let Some(message) = negotiate_error {
            app.negotiate_running = false;
            app.negotiate_boundary = None;
            app.status = format!("negotiate error: {}", message);
            app.push_log("negotiate error");
            stop_negotiate_process(&mut negotiate_proc);
            if app.negotiate_batch.is_some() {
                app.negotiate_batch = None;
                app.push_log("negotiation batch stopped");
            }
        }
        let mut negotiate_exit: Option<bool> = None;
        if let Some(proc) = negotiate_proc.as_mut() {
            if let Ok(Some(exit)) = proc.child.try_wait() {
                if exit.success() {
                    app.negotiate_running = false;
                    app.negotiate_boundary = None;
                    app.refresh_boundaries();
                    app.status = format!("negotiation done | {} rounds", app.negotiate_round);
                    app.push_log("negotiation complete");
                    negotiate_exit = Some(true);
                } else {
                    let mut stderr = String::new();
                    if let Some(mut err) = proc.child.stderr.take() {
                        let _ = err.read_to_string(&mut stderr);
                    }
                    app.negotiate_running = false;
                    app.negotiate_boundary = None;
                    app.status = format!("negotiate error: {} {}", exit, stderr);
                    app.push_log("negotiate error");
                    negotiate_exit = Some(false);
                }
            }
        }
        if let Some(success) = negotiate_exit {
            stop_negotiate_process(&mut negotiate_proc);
            if success {
                advance_negotiate_batch(&mut app, &mut negotiate_proc);
            } else if app.negotiate_batch.is_some() {
                app.negotiate_batch = None;
                app.push_log("negotiation batch stopped");
            }
        }
        if let Some(rx) = process_rx.as_ref() {
            let mut process_done = false;
            let mut process_error: Option<String> = None;
            while let Ok(event) = rx.try_recv() {
                match event {
                    ProcessEvent::Update { index, start, end, score } => {
                        if index < app.chunks.len() {
                            let span = SegmentSpan { start, end, score };
                            app.chunks[index] = build_chunk_from_span(index, span, &app.text, app.boundary_cutoff);
                            app.active_index = Some(index);
                            app.flash_until = Some(Instant::now() + Duration::from_millis(300));
                            app.record_gpu_op();
                            app.refresh_boundaries();
                            if index % 5 == 0 {
                                app.push_log(&format!("process update chunk {}", index + 1));
                            }
                        }
                    }
                    ProcessEvent::Log { message } => {
                        app.push_log(&message);
                    }
                    ProcessEvent::Done => {
                        process_done = true;
                    }
                    ProcessEvent::Error { message } => {
                        process_error = Some(message);
                    }
                }
            }
            if process_done {
                app.process_running = false;
                app.status = "process done".to_string();
                app.push_log("process done");
                process_rx = None;
            } else if let Some(message) = process_error {
                app.process_running = false;
                app.status = format!("process error: {}", message);
                app.push_log("process error");
                process_rx = None;
            }
        }
        if app.processing_pending && !app.processing && !app.chunks.is_empty() {
            if !app.score_config.is_embed_enabled() {
                app.processing_pending = false;
                app.status = "vec2text embed script missing".to_string();
            } else {
                app.reset_processing_state();
                app.processing = true;
                app.processing_pending = false;
                app.embed_started_at = Some(Instant::now());
                app.last_embed_log = Some(Instant::now());
                app.pending_embeddings.clear();
                app.embed_done_pending = false;
                app.last_embed_tick = Instant::now();
                let texts: Vec<String> = app.chunks.iter().map(|c| c.text.clone()).collect();
                let total = texts.len();
                let batches = (total + app.score_config.embed_batch_size - 1) / app.score_config.embed_batch_size;
                match spawn_embedder(texts, &app.score_config) {
                    Ok(rx) => {
                        embed_rx = Some(rx);
                        app.status = format!(
                            "embedding 0/{} | window {} tokens",
                            app.chunks.len(),
                            app.window_tokens
                        );
                        app.push_log(&format!(
                            "embedding {} chunks ({} batches)",
                            total,
                            batches
                        ));
                    }
                    Err(err) => {
                        app.processing = false;
                        app.status = format!("embedder start failed: {}", err);
                    }
                }
            }
        }
        if app.auto_embed_pending && !app.processing && !app.process_running && !app.chunks.is_empty() {
            if app.score_config.is_embed_enabled() {
                app.auto_embed_pending = false;
                app.processing = true;
                app.embed_started_at = Some(Instant::now());
                app.last_embed_log = Some(Instant::now());
                app.pending_embeddings.clear();
                app.embed_done_pending = false;
                app.last_embed_tick = Instant::now();
                let texts: Vec<String> = app.chunks.iter().map(|c| c.text.clone()).collect();
                let total = texts.len();
                let batches = (total + app.score_config.embed_batch_size - 1) / app.score_config.embed_batch_size;
                match spawn_embedder(texts, &app.score_config) {
                    Ok(rx) => {
                        embed_rx = Some(rx);
                        app.status = format!(
                            "auto-embedding 0/{} | window {} tokens",
                            app.chunks.len(),
                            app.window_tokens
                        );
                        app.push_log(&format!(
                            "auto-embedding {} chunks ({} batches)",
                            total,
                            batches
                        ));
                    }
                    Err(err) => {
                        app.processing = false;
                        app.status = format!("auto-embed failed: {}", err);
                    }
                }
            } else {
                app.auto_embed_pending = false;
            }
        }
        let mut embed_error: Option<String> = None;
        let mut embed_done = false;
        if let Some(rx) = &embed_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    EmbedEvent::Embedding { index, embedding } => {
                        if index < app.embeddings.len() {
                            app.pending_embeddings.push_back((index, embedding));
                            app.record_gpu_op();
                        }
                    }
                    EmbedEvent::Log { message } => {
                        app.push_log(&message);
                    }
                    EmbedEvent::Done => {
                        embed_done = true;
                    }
                    EmbedEvent::Error { message } => {
                        embed_error = Some(message);
                    }
                }
            }
        }
        if embed_done {
            app.embed_done_pending = true;
            embed_rx = None;
        }
        if app.processing && !app.pending_embeddings.is_empty() {
            if app.last_embed_tick.elapsed() >= Duration::from_millis(80) {
                if let Some((index, embedding)) = app.pending_embeddings.pop_front() {
                    if index < app.embeddings.len() {
                        if app.embeddings[index].is_none() {
                            app.processed_count += 1;
                        }
                        app.embeddings[index] = Some(embedding);
                        app.active_index = Some(index);
                        app.flash_until = Some(Instant::now() + Duration::from_millis(350));
                        app.update_highlights(index);
                        app.status = format!(
                            "embedded {}/{} | active {}",
                            app.processed_count,
                            app.chunks.len(),
                            index + 1
                        );
                        if index % 5 == 0 {
                            app.push_log(&format!(
                                "embedded chunk {}/{}",
                                index + 1,
                                app.chunks.len()
                            ));
                        }
                    }
                }
                app.last_embed_tick = Instant::now();
            }
        }
        if app.embed_done_pending && app.pending_embeddings.is_empty() && app.processing {
            app.processing = false;
            app.active_index = None;
            app.flash_until = None;
            app.processing_pending = false;
            app.embed_started_at = None;
            app.last_embed_log = None;
            app.embed_done_pending = false;
            app.push_log("embedding done");

            // Auto-start processor to compute scores
            if app.score_config.is_process_enabled() && !app.chunks.is_empty() {
                app.process_running = true;
                app.push_log("auto-starting processor");
                match spawn_processer(app.text.clone(), &app.chunks, &app.score_config) {
                    Ok(rx) => {
                        process_rx = Some(rx);
                        app.status = format!("processing {} chunks", app.chunks.len());
                    }
                    Err(e) => {
                        app.process_running = false;
                        app.status = format!("process error: {}", e);
                        app.push_log(&format!("process spawn error: {}", e));
                    }
                }
            } else {
                app.status = format!("embedding done | {} chunks", app.chunks.len());
            }
        }
        if let Some(message) = embed_error {
            app.processing = false;
            app.active_index = None;
            app.flash_until = None;
            app.processing_pending = false;
            app.embed_started_at = None;
            app.last_embed_log = None;
            app.embed_done_pending = false;
            app.pending_embeddings.clear();
            app.status = format!("embedder error: {}", message);
            app.push_log("embedder error");
            embed_rx = None;
        }
        terminal.draw(|f| draw_ui(f, &mut app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                // Only handle key press, ignore release/repeat to avoid double triggers
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Esc => {
                        // Close menu, clear dimming, or stop processing
                        if app.show_menu {
                            app.show_menu = false;
                            app.status = "menu closed".to_string();
                        } else if app.dim_dissimilar {
                            app.dim_dissimilar = false;
                            app.similar_chunks.clear();
                            app.status = "similarity view cleared".to_string();
                        } else if app.negotiate_running {
                            app.negotiate_running = false;
                            app.negotiate_batch = None;
                            stop_negotiate_process(&mut negotiate_proc);
                            app.status = "negotiation stopped".to_string();
                        } else if app.processing {
                            app.reset_processing_state();
                            embed_rx = None;
                            app.status = "processing stopped".to_string();
                        }
                    }
                    KeyCode::Up => {
                        if app.show_menu {
                            // Navigate menu
                            if app.menu_selection > 0 {
                                app.menu_selection -= 1;
                            }
                        } else {
                            app.select_prev();
                        }
                    }
                    KeyCode::Down => {
                        if app.show_menu {
                            // Navigate menu
                            if app.menu_selection + 1 < MENU_ITEMS.len() {
                                app.menu_selection += 1;
                            }
                        } else {
                            app.select_next();
                        }
                    }
                    KeyCode::PageUp => {
                        if app.show_similarity && app.focus_left {
                            app.similarity_scroll = app.similarity_scroll.saturating_sub(10);
                            app.similarity_selected = app.similarity_selected.min(
                                app.similarity_scroll + 19  // keep selected in view
                            ).max(app.similarity_scroll);
                        } else {
                            // Scroll source overview
                            app.source_scroll = app.source_scroll.saturating_sub(10);
                        }
                    }
                    KeyCode::PageDown => {
                        if app.show_similarity && app.focus_left {
                            let max_scroll = app.similar_chunks.len().saturating_sub(10);
                            app.similarity_scroll = (app.similarity_scroll + 10).min(max_scroll);
                            app.similarity_selected = app.similarity_selected.max(app.similarity_scroll);
                        } else {
                            // Scroll source overview (limit will be applied in draw)
                            app.source_scroll += 10;
                        }
                    }
                    KeyCode::Home => {
                        if app.show_similarity && app.focus_left {
                            app.similarity_scroll = 0;
                            app.similarity_selected = 0;
                        } else {
                            app.source_scroll = 0;
                        }
                    }
                    KeyCode::End => {
                        if app.show_similarity && app.focus_left {
                            let max = app.similar_chunks.len().saturating_sub(1);
                            app.similarity_scroll = max.saturating_sub(10);
                            app.similarity_selected = max;
                        } else {
                            app.source_scroll = usize::MAX; // will be clamped in draw
                        }
                    }
                    KeyCode::Left => {
                        if app.show_similarity && app.focus_left && app.similarity_scroll > 0 {
                            app.similarity_scroll -= 1;
                            if app.similarity_selected > 0 {
                                app.similarity_selected -= 1;
                            }
                        } else if !app.show_similarity && app.source_scroll > 0 {
                            app.source_scroll -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if app.show_similarity && app.focus_left {
                            let max_scroll = app.similar_chunks.len().saturating_sub(10);
                            app.similarity_scroll = (app.similarity_scroll + 1).min(max_scroll);
                            if app.similarity_selected < app.similar_chunks.len().saturating_sub(1) {
                                app.similarity_selected += 1;
                            }
                        } else if !app.show_similarity {
                            app.source_scroll += 1;
                        }
                    }
                    KeyCode::Enter => {
                        if app.show_menu {
                            // Execute menu selection
                            match app.menu_selection {
                                0 => {
                                    // Negotiate boundaries
                                    app.show_menu = false;
                                    if app.negotiate_batch.is_some() {
                                        app.status = "negotiation batch running".to_string();
                                        app.push_log("negotiation batch running");
                                    } else if app.negotiate_running {
                                        app.status = "negotiation already running".to_string();
                                        app.push_log("negotiation already running");
                                    } else if app.chunks.len() < 2 {
                                        app.status = "need at least 2 chunks".to_string();
                                        app.push_log("need at least 2 chunks to negotiate");
                                    } else if !app.score_config.is_negotiate_enabled() {
                                        app.status = "negotiate script not found".to_string();
                                        app.push_log("negotiate script not found");
                                    } else {
                                        let selected = app.selected;
                                        match start_negotiation(&mut app, &mut negotiate_proc, selected) {
                                            Ok(()) => {
                                                app.status = format!("chunk {} negotiating boundaries...", app.selected + 1);
                                                app.push_log(&format!(
                                                    "chunk {} asking: what can I give up?",
                                                    app.selected + 1
                                                ));
                                            }
                                            Err(err) => {
                                                app.status = format!("negotiate failed: {}", err);
                                                app.push_log("negotiate failed to start");
                                            }
                                        }
                                    }
                                }
                                1 => {
                                    // Show similar (dim dissimilar chunks in-place)
                                    app.show_menu = false;
                                    if app.embeddings.iter().any(|e| e.is_some()) {
                                        // Compute similarities to selected chunk
                                        if let Some(Some(selected_embed)) = app.embeddings.get(app.selected) {
                                            app.similar_chunks.clear();
                                            for (idx, embed_opt) in app.embeddings.iter().enumerate() {
                                                if let Some(embed) = embed_opt {
                                                    let sim = cosine_similarity(selected_embed, embed);
                                                    app.similar_chunks.push((idx, sim));
                                                }
                                            }
                                            app.similar_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                            app.dim_dissimilar = true;
                                            app.status = format!("showing similarity to chunk {} - Esc to clear", app.selected + 1);
                                        }
                                    } else {
                                        app.status = "no embeddings - run embed first".to_string();
                                    }
                                }
                                2 => {
                                    // Negotiate all chunks
                                    app.show_menu = false;
                                    start_negotiate_all(&mut app, &mut negotiate_proc);
                                }
                                _ => {}
                            }
                        } else {
                            // Open menu for selected chunk
                            app.show_menu = true;
                            app.menu_selection = 0;
                            app.status = format!("chunk {} - select action", app.selected + 1);
                        }
                    }
                    KeyCode::Tab => {
                        // Cycle focus: ChunkList -> Search -> SourceText -> ChunkList
                        app.focus_area = match app.focus_area {
                            FocusArea::ChunkList => FocusArea::Search,
                            FocusArea::Search => FocusArea::SourceText,
                            FocusArea::SourceText => FocusArea::ChunkList,
                        };
                        app.status = match app.focus_area {
                            FocusArea::Search => "search mode - type to search".to_string(),
                            FocusArea::SourceText => "source view - scroll with arrows".to_string(),
                            FocusArea::ChunkList => "chunk list - navigate with arrows".to_string(),
                        };
                    }
                    KeyCode::Backspace => {
                        if app.focus_area == FocusArea::Search && !app.search_query.is_empty() {
                            app.search_query.pop();
                            app.update_search();
                        }
                    }
                    KeyCode::Char(c) => {
                        if app.focus_area == FocusArea::Search {
                            app.search_query.push(c);
                            app.update_search();
                        } else {
                            // Handle other char keys (existing functionality)
                            match c {
                                'q' => {
                                    if app.expanded_chunk.is_some() {
                                        app.expanded_chunk = None;
                                    } else if app.show_similarity {
                                        app.show_similarity = false;
                                        app.focus_left = false;
                                    } else {
                                        break;
                                    }
                                }
                                'r' => {
                                    embed_rx = None;
                                    process_rx = None;
                                    segment_rx = None;
                                    invert_rx = None;
                                    explore_rx = None;
                                    stop_negotiate_process(&mut negotiate_proc);
                                    app.negotiate_running = false;
                                    app.negotiate_batch = None;
                                    app.reset_processing_state();
                                    let _ = app.reload();
                                }
                                '+' | '=' => {
                                    embed_rx = None;
                                    process_rx = None;
                                    segment_rx = None;
                                    invert_rx = None;
                                    explore_rx = None;
                                    stop_negotiate_process(&mut negotiate_proc);
                                    app.negotiate_running = false;
                                    app.negotiate_batch = None;
                                    app.reset_processing_state();
                                    app.adjust_window(4);
                                }
                                '-' => {
                                    embed_rx = None;
                                    process_rx = None;
                                    segment_rx = None;
                                    invert_rx = None;
                                    explore_rx = None;
                                    stop_negotiate_process(&mut negotiate_proc);
                                    app.negotiate_running = false;
                                    app.negotiate_batch = None;
                                    app.reset_processing_state();
                                    app.adjust_window(-4);
                                }
                                'g' => {
                                    if segment_rx.is_some() {
                                        app.push_log("segmenter already running");
                                        app.status = "segmenter already running".to_string();
                                    } else if app.score_config.is_vec2text_enabled() {
                                        segment_rx = spawn_segmenter(app.text.clone(), &app.score_config).ok();
                                        app.push_log("segmenter started");
                                        app.status = "segmenter running".to_string();
                                    } else {
                                        app.push_log("vec2text scripts not found");
                                        app.status = "vec2text scripts not found".to_string();
                                    }
                                }
                                's' => {
                                    if app.processing {
                                        app.processing = false;
                                        app.active_index = None;
                                        app.flash_until = None;
                                        app.processing_pending = false;
                                        app.embed_started_at = None;
                                        app.last_embed_log = None;
                                        app.pending_embeddings.clear();
                                        app.embed_done_pending = false;
                                        embed_rx = None;
                                        app.status = "processing stopped".to_string();
                                        app.push_log("processing stopped");
                                    } else if app.process_running {
                                        app.process_running = false;
                                        process_rx = None;
                                        app.status = "process stopped".to_string();
                                        app.push_log("process stopped");
                                    } else if app.invert_running {
                                        app.invert_running = false;
                                        invert_rx = None;
                                        app.status = "inversion stopped".to_string();
                                        app.push_log("inversion stopped");
                                    } else if app.explore_running {
                                        app.explore_running = false;
                                        explore_rx = None;
                                        app.status = "exploration stopped".to_string();
                                        app.push_log("exploration stopped");
                                    } else if app.negotiate_running {
                                        app.negotiate_running = false;
                                        app.negotiate_batch = None;
                                        stop_negotiate_process(&mut negotiate_proc);
                                        app.status = "negotiation stopped".to_string();
                                        app.push_log("negotiation stopped");
                                    } else {
                                        app.status = "not processing".to_string();
                                        app.push_log("not processing");
                                    }
                                }
                                'p' => {
                                    if app.process_running {
                                        app.status = "process already running".to_string();
                                        app.push_log("process already running");
                                    } else if app.chunks.is_empty() {
                                        if segment_rx.is_some() {
                                            app.processing_pending = true;
                                            app.status = "waiting for segmentation...".to_string();
                                            app.push_log("waiting for segmentation...");
                                        } else {
                                            app.status = "no chunks to process".to_string();
                                            app.push_log("no chunks available");
                                        }
                                    } else if !app.score_config.is_process_enabled() {
                                        app.status = "vec2text process script missing".to_string();
                                        app.push_log("process script missing");
                                    } else {
                                        embed_rx = None;
                                        app.processing = false;
                                        app.processing_pending = false;
                                        app.auto_embed_pending = false;
                                        app.pending_embeddings.clear();
                                        app.embed_done_pending = false;
                                        app.process_running = true;
                                        app.active_index = None;
                                        app.flash_until = None;
                                        app.push_log("process start: directional slide");
                                        segment_rx = None;
                                        match spawn_processer(app.text.clone(), &app.chunks, &app.score_config) {
                                            Ok(rx) => {
                                                process_rx = Some(rx);
                                                app.status = format!(
                                                    "process running | {} chunks",
                                                    app.chunks.len()
                                                );
                                            }
                                            Err(err) => {
                                                app.process_running = false;
                                                app.status = format!("process start failed: {}", err);
                                                app.push_log("process start failed");
                                            }
                                        }
                                    }
                                }
                                'i' => {
                                    if app.invert_running {
                                        app.push_log("inversion already running");
                                        app.status = "inversion already running".to_string();
                                    } else if app.chunks.is_empty() {
                                        app.push_log("no chunks to invert");
                                        app.status = "no chunks to invert".to_string();
                                    } else if !app.score_config.is_invert_enabled() {
                                        app.push_log("invert script not found");
                                        app.status = "invert script not found".to_string();
                                    } else {
                                        app.invert_running = true;
                                        app.inversions = vec![None; app.chunks.len()];
                                        let texts: Vec<String> = app.chunks.iter().map(|c| c.text.clone()).collect();
                                        match spawn_inverter(texts, &app.score_config) {
                                            Ok(rx) => {
                                                invert_rx = Some(rx);
                                                app.status = format!("inverting {} chunks...", app.chunks.len());
                                                app.push_log(&format!("starting inversion of {} chunks", app.chunks.len()));
                                            }
                                            Err(err) => {
                                                app.invert_running = false;
                                                app.status = format!("invert failed: {}", err);
                                                app.push_log("invert failed to start");
                                            }
                                        }
                                    }
                                }
                                'v' => {
                                    if app.inversions.iter().any(|i| i.is_some()) {
                                        app.show_inversions = !app.show_inversions;
                                        app.status = if app.show_inversions {
                                            "showing inversions".to_string()
                                        } else {
                                            "showing original text".to_string()
                                        };
                                    } else {
                                        app.push_log("no inversions yet - press 'i' to invert");
                                        app.status = "no inversions - press 'i' first".to_string();
                                    }
                                }
                                'm' => {
                                    if app.explore_running {
                                        app.push_log("exploration already running");
                                        app.status = "exploration already running".to_string();
                                    } else if app.text.is_empty() {
                                        app.push_log("no text to explore");
                                        app.status = "no text to explore".to_string();
                                    } else if !app.score_config.is_explore_enabled() {
                                        app.push_log("explore script not found");
                                        app.status = "explore script not found".to_string();
                                    } else {
                                        app.explore_running = true;
                                        app.explored_spans.clear();
                                        app.explore_sample_count = 0;
                                        let num_samples = 100;
                                        match spawn_explorer(app.text.clone(), &app.score_config, num_samples) {
                                            Ok(rx) => {
                                                explore_rx = Some(rx);
                                                app.status = format!("Monte Carlo: sampling {} spans...", num_samples);
                                                app.push_log(&format!("starting Monte Carlo exploration: {} samples", num_samples));
                                            }
                                            Err(err) => {
                                                app.explore_running = false;
                                                app.status = format!("explore failed: {}", err);
                                                app.push_log("explore failed to start");
                                            }
                                        }
                                    }
                                }
                                'n' => {
                                    if app.negotiate_batch.is_some() {
                                        app.push_log("negotiation batch running");
                                        app.status = "negotiation batch running".to_string();
                                    } else if app.negotiate_running {
                                        app.push_log("negotiation already running");
                                        app.status = "negotiation already running".to_string();
                                    } else if app.chunks.len() < 2 {
                                        app.push_log("need at least 2 chunks to negotiate");
                                        app.status = "need at least 2 chunks".to_string();
                                    } else if !app.score_config.is_negotiate_enabled() {
                                        app.push_log("negotiate script not found");
                                        app.status = "negotiate script not found".to_string();
                                    } else {
                                        let selected = app.selected;
                                        match start_negotiation(&mut app, &mut negotiate_proc, selected) {
                                            Ok(()) => {
                                                app.status = format!("chunk {} negotiating boundaries...", app.selected + 1);
                                                app.push_log(&format!(
                                                    "chunk {} asking: what can I give up?",
                                                    app.selected + 1
                                                ));
                                            }
                                            Err(err) => {
                                                app.status = format!("negotiate failed: {}", err);
                                                app.push_log("negotiate failed to start");
                                            }
                                        }
                                    }
                                }
                                'N' => {
                                    start_negotiate_all(&mut app, &mut negotiate_proc);
                                }
                                'h' => {
                                    // Toggle similarity view for selected chunk
                                    if app.show_similarity {
                                        app.show_similarity = false;
                                        app.similar_chunks.clear();
                                        app.similarity_scroll = 0;
                                        app.similarity_selected = 0;
                                        app.focus_left = false;
                                        app.expanded_chunk = None;
                                        app.status = "similarity view closed".to_string();
                                    } else {
                                        if app.embeddings.get(app.selected).and_then(|e| e.as_ref()).is_some() {
                                            app.compute_similarity_ranking();
                                            app.show_similarity = true;
                                            app.focus_left = true;
                                            app.similarity_selected = 0;
                                            app.status = format!("showing similarity for chunk {} ({} similar) - Tab to switch panels",
                                                app.selected + 1, app.similar_chunks.len());
                                        } else {
                                            app.push_log("no embedding for selected chunk - run embedding first");
                                            app.status = "no embedding - press keys to embed first".to_string();
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
            app.update_processing_status();
            if app.processing && app.processed_count == 0 {
                if let (Some(start), Some(last)) = (app.embed_started_at, app.last_embed_log) {
                    if last.elapsed() >= Duration::from_secs(5) {
                        let elapsed = start.elapsed().as_secs();
                        app.push_log(&format!("still loading model... {}s", elapsed));
                        app.last_embed_log = Some(Instant::now());
                    }
                }
            }
        }
    }

    stop_negotiate_process(&mut negotiate_proc);
    if let Some(mut server) = app.model_server.take() {
        stop_child_process(&mut server.child);
    }

    restore_terminal(terminal)
}

fn draw_ui(f: &mut ratatui::Frame, app: &mut App) {
    let size = f.size();

    // Check if we're in expanded view mode - render full-width
    if let Some(chunk_idx) = app.expanded_chunk {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(size);

        // Header
        let chunk = app.chunks.get(chunk_idx);
        let (score, start, end) = chunk.map(|c| (c.score, c.start, c.end)).unwrap_or((0.0, 0, 0));
        let similarity = app.similar_chunks.iter()
            .find(|(idx, _)| *idx == chunk_idx)
            .map(|(_, s)| *s);
        let sim_str = similarity.map(|s| format!(" | similarity: {:.3}", s)).unwrap_or_default();

        let title = Line::from(vec![
            Span::styled("Chunk ", Style::default().fg(Color::Cyan)),
            Span::styled(format!("{}", chunk_idx + 1), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(format!("  [{}-{}]  score: {:.3}{}", start, end, score, sim_str)),
        ]);
        let header = Paragraph::new(Text::from(title))
            .block(Block::default().borders(Borders::ALL).title("Expanded View"));
        f.render_widget(header, layout[0]);

        // Full text content
        let full_text = chunk.map(|c| c.text.clone()).unwrap_or_default();
        let content = Paragraph::new(full_text)
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                "Full Text (Enter/Esc to close)",
                Style::default().fg(Color::Green)
            )))
            .wrap(Wrap { trim: false });
        f.render_widget(content, layout[1]);

        // Help
        let help = Paragraph::new("Enter/Esc close  q close")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(help, layout[2]);
        return;
    }

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(size);

    // Header with logo, search box, and status
    let search_style = if app.focus_area == FocusArea::Search {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().fg(Color::Gray)
    };
    let search_display = if app.search_query.is_empty() && app.focus_area != FocusArea::Search {
        "search...".to_string()
    } else {
        format!("{}_", app.search_query)
    };

    let title = Line::from(vec![
        Span::styled(
            "🚀 Vector's Mite",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled("[", Style::default().fg(Color::DarkGray)),
        Span::styled(search_display, search_style),
        Span::styled("]", Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(&app.status, Style::default().fg(Color::Yellow)),
    ]);
    let header = Paragraph::new(Text::from(title))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, layout[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[1]);

    if app.show_similarity {
        // Render similarity view instead of source overview
        let visible_height = body[0].height.saturating_sub(2) as usize; // account for borders
        let similar_lines: Vec<Line> = app.similar_chunks
            .iter()
            .skip(app.similarity_scroll)
            .take(visible_height)
            .enumerate()
            .map(|(vis_idx, (chunk_idx, score))| {
                let abs_idx = app.similarity_scroll + vis_idx;
                let rank = abs_idx + 1;
                let chunk = app.chunks.get(*chunk_idx);
                let preview = chunk.map(|c| c.preview.clone()).unwrap_or_default();
                let is_source = *chunk_idx == app.selected;
                let is_selected = abs_idx == app.similarity_selected && app.focus_left;

                // Color based on similarity score
                let score_color = if *score > 0.9 {
                    Color::Green
                } else if *score > 0.7 {
                    Color::Yellow
                } else if *score > 0.5 {
                    Color::Cyan
                } else {
                    Color::Gray
                };

                let style = if is_source {
                    Style::default().fg(Color::White).bg(Color::Blue).add_modifier(Modifier::BOLD)
                } else if is_selected {
                    // Highlight the selected item
                    Style::default().fg(Color::Black).bg(score_color).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(score_color)
                };

                // Selection indicator
                let selector = if is_selected { "▶ " } else { "  " };

                Line::from(vec![
                    Span::styled(selector.to_string(), Style::default().fg(Color::White)),
                    Span::styled(format!("{:>3}. ", rank), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("[{:>3}] ", chunk_idx + 1), style),
                    Span::styled(format!("{:.3} ", score), Style::default().fg(score_color).add_modifier(Modifier::BOLD)),
                    Span::styled(preview.chars().take(55).collect::<String>(), style),
                ])
            })
            .collect();

        let source_chunk = app.chunks.get(app.selected);
        let source_preview = source_chunk.map(|c| c.preview.chars().take(25).collect::<String>()).unwrap_or_default();
        let focus_indicator = if app.focus_left { " [FOCUS]" } else { "" };
        let title = format!("Similar to #{} \"{}...\"{} (Tab=switch, Enter=expand, j=jump)",
            app.selected + 1, source_preview, focus_indicator);

        let scroll_info = format!(" [{}-{}/{}]",
            app.similarity_scroll + 1,
            (app.similarity_scroll + visible_height).min(app.similar_chunks.len()),
            app.similar_chunks.len());

        let similarity_view = Paragraph::new(Text::from(similar_lines))
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!("{}{}", title, scroll_info),
                Style::default().fg(Color::Cyan)
            )));
        f.render_widget(similarity_view, body[0]);
    } else if app.show_inversions {
        // Render inversions view - what each chunk "says" according to the model
        let wrap_at = body[0].width.saturating_sub(4) as usize;
        let wrap_at = wrap_at.max(1);

        let mut all_lines: Vec<Line> = Vec::new();

        for (idx, _chunk) in app.chunks.iter().enumerate() {
            let inversion = app.inversions.get(idx).and_then(|inv| inv.as_ref());
            let is_selected = idx == app.selected;

            match inversion {
                Some(inv) => {
                    // Color based on round-trip score
                    let score_color = if inv.score > 0.9 {
                        Color::Green
                    } else if inv.score > 0.7 {
                        Color::Yellow
                    } else if inv.score > 0.5 {
                        Color::Cyan
                    } else {
                        Color::Red
                    };

                    let selector = if is_selected { "▶" } else { " " };
                    let header_style = if is_selected {
                        Style::default().fg(Color::Black).bg(score_color).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(score_color).add_modifier(Modifier::BOLD)
                    };

                    // Header line: selector, chunk number, score
                    all_lines.push(Line::from(vec![
                        Span::styled(format!("{} ", selector), Style::default().fg(Color::White)),
                        Span::styled(format!("[{:>3}] ", idx + 1), header_style),
                        Span::styled(format!("score: {:.3}", inv.score), Style::default().fg(score_color)),
                    ]));

                    // Content: wrap the inverted text using our helper
                    let wrapped = wrap_text(&inv.inverted, wrap_at);
                    for line in wrapped.lines() {
                        all_lines.push(Line::from(vec![
                            Span::styled("  ", Style::default()),
                            Span::styled(line.to_string(), Style::default().fg(score_color)),
                        ]));
                    }
                    // Handle empty text
                    if inv.inverted.is_empty() {
                        all_lines.push(Line::from(vec![
                            Span::styled("  ", Style::default()),
                            Span::styled("(empty)", Style::default().fg(Color::DarkGray)),
                        ]));
                    }

                    // Blank line between chunks
                    all_lines.push(Line::from(""));
                }
                None => {
                    let selector = if is_selected { "▶" } else { " " };
                    all_lines.push(Line::from(vec![
                        Span::styled(format!("{} ", selector), Style::default().fg(Color::White)),
                        Span::styled(format!("[{:>3}] ", idx + 1), Style::default().fg(Color::DarkGray)),
                        Span::styled("... pending ...", Style::default().fg(Color::DarkGray)),
                    ]));
                    all_lines.push(Line::from(""));
                }
            }
        }

        let total_lines = all_lines.len();
        let visible_height = body[0].height.saturating_sub(2) as usize;
        let max_scroll = total_lines.saturating_sub(visible_height);
        let effective_scroll = app.source_scroll.min(max_scroll);
        let scroll_offset = effective_scroll.min(u16::MAX as usize) as u16;

        let inverted_count = app.inversions.iter().filter(|i| i.is_some()).count();
        let scroll_info = if total_lines > visible_height {
            format!(" [{}-{}/{}]", effective_scroll + 1, (effective_scroll + visible_height).min(total_lines), total_lines)
        } else {
            String::new()
        };

        let inversion_view = Paragraph::new(Text::from(all_lines))
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!("Inversions ({}/{} done){}", inverted_count, app.chunks.len(), scroll_info),
                Style::default().fg(Color::Magenta)
            )))
            .scroll((scroll_offset, 0));
        f.render_widget(inversion_view, body[0]);
    } else {
        let inner_width = body[0].width.saturating_sub(2) as usize;
        let visible_height = body[0].height.saturating_sub(2) as usize;
        let selected_marker_index = {
            let mut chunks = app.chunks.clone();
            chunks.sort_by_key(|chunk| chunk.start);
            chunks.iter().position(|chunk| chunk.index == app.selected)
        };

        // First pass: calculate line count and selected line for scroll adjustment
        let (line_count, selected_line) = {
            let overview_lines = build_overview_lines(app);
            let (wrapped, sel) = wrap_lines_with_marker(
                overview_lines,
                inner_width.max(1),
                selected_marker_index,
            );
            (wrapped.len(), sel)
        };

        let max_scroll = line_count.saturating_sub(visible_height);
        let new_scroll = if app.focus_area != FocusArea::SourceText {
            if let Some(target_line) = selected_line {
                let desired = target_line.saturating_sub(visible_height / 3);
                desired.min(max_scroll)
            } else {
                app.source_scroll.min(max_scroll)
            }
        } else {
            app.source_scroll.min(max_scroll)
        };
        let scroll_offset = new_scroll.min(u16::MAX as usize) as u16;

        {
            let overview_lines = build_overview_lines(app);
            let (wrapped_lines, _) = wrap_lines_with_marker(
                overview_lines,
                inner_width.max(1),
                selected_marker_index,
            );

            let overview = Paragraph::new(Text::from(wrapped_lines))
                .block(Block::default().borders(Borders::ALL).title("Source Text"))
                .scroll((scroll_offset, 0));
            f.render_widget(overview, body[0]);
        }
        app.source_scroll = new_scroll;
    }

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3), Constraint::Length(6)])
        .split(body[1]);

    let items: Vec<ListItem> = app
        .chunks
        .iter()
        .map(|chunk| {
            let line = format!(
                "{:>3} [{:>5}-{:>5}] q={:.3} {}",
                chunk.index + 1,
                chunk.start,
                chunk.end,
                chunk.score,
                chunk.preview
            );
            ListItem::new(line).style(list_style(app, chunk))
        })
        .collect();
    let chunks_title = if app.show_similarity && !app.focus_left {
        Span::styled("Chunks [FOCUS]", Style::default().fg(Color::Yellow))
    } else if app.show_similarity {
        Span::styled("Chunks (Tab to focus)", Style::default().fg(Color::DarkGray))
    } else {
        Span::raw("Chunks")
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(chunks_title))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, right[0], &mut app.list_state);

    // GPU throughput bar graph
    let bar_width = right[1].width.saturating_sub(2) as usize; // account for borders
    let max_ops = 20.0f32; // scale: 20 ops/sec = full bar
    let fill_ratio = (app.gpu_ops_per_sec / max_ops).min(1.0);
    let filled = (bar_width as f32 * fill_ratio) as usize;
    let empty = bar_width.saturating_sub(filled);
    let bar_color = if app.gpu_ops_per_sec > 10.0 {
        Color::Green
    } else if app.gpu_ops_per_sec > 3.0 {
        Color::Yellow
    } else if app.gpu_ops_per_sec > 0.0 {
        Color::Red
    } else {
        Color::DarkGray
    };
    let bar_line = Line::from(vec![
        Span::styled("█".repeat(filled), Style::default().fg(bar_color)),
        Span::styled("░".repeat(empty), Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" {:.1}/s", app.gpu_ops_per_sec)),
    ]);
    let gpu_bar = Paragraph::new(bar_line)
        .block(Block::default().borders(Borders::ALL).title("GPU ops"));
    f.render_widget(gpu_bar, right[1]);

    let log_lines: Vec<Line> = if app.logs.is_empty() {
        vec![Line::from("log idle")]
    } else {
        app.logs.iter().map(|line| Line::from(line.as_str())).collect()
    };
    let log = Paragraph::new(Text::from(log_lines))
        .block(Block::default().borders(Borders::ALL).title("Embed log"))
        .wrap(Wrap { trim: false });
    f.render_widget(log, right[2]);

    let help_text = if app.show_similarity {
        if app.focus_left {
            "Tab=switch  ↑/↓=select  Enter=expand  j=jump  h/Esc=close  PgUp/Dn=page"
        } else {
            "Tab=switch  ↑/↓=nav  Enter=expand  h/Esc=close  PgUp/Dn=scroll"
        }
    } else if app.show_inversions {
        "v=close  ↑/↓=nav  PgUp/Dn=scroll  i=run inversion  green=good yellow=ok red=poor"
    } else {
        "q quit  r reload  ↑/↓ nav  PgUp/Dn scroll  n/N negotiate  Enter menu  Esc close"
    };
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray));
    f.render_widget(help, layout[2]);

    // Render menu popup if open
    if app.show_menu {
        let menu_height = MENU_ITEMS.len() as u16 + 2; // +2 for borders
        let menu_width = 25u16;

        // Center the menu
        let area = f.size();
        let menu_x = (area.width.saturating_sub(menu_width)) / 2;
        let menu_y = (area.height.saturating_sub(menu_height)) / 2;
        let menu_area = ratatui::layout::Rect::new(menu_x, menu_y, menu_width, menu_height);

        // Clear background
        f.render_widget(ratatui::widgets::Clear, menu_area);

        let items: Vec<ListItem> = MENU_ITEMS.iter().enumerate().map(|(i, item)| {
            let style = if i == app.menu_selection {
                Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(format!(" {} ", item), style)))
        }).collect();

        let menu = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    format!(" Chunk {} ", app.selected + 1),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                ))
                .border_style(Style::default().fg(Color::Cyan)));
        f.render_widget(menu, menu_area);
    }
}
