use anyhow::{anyhow, Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
    Terminal,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::{self, BufRead, Read, Stdout, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use unicode_width::UnicodeWidthChar;

const DEFAULT_WINDOW_TOKENS: usize = 32;
const DEFAULT_EMBED_BATCH_SIZE: usize = 64;
const DEFAULT_EVENTS_PORT: u16 = 7011;
const MAX_LOG_LINES: usize = 6;
const BLINK_FRAGMENT_LIMIT: usize = 200;
const EVENT_DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vector's Might Blink Monitor</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin: 24px; }
    header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    button { padding: 8px 12px; font-weight: 600; }
    .status { font-size: 12px; opacity: 0.7; }
    .row { display: grid; grid-template-columns: 120px 1fr; gap: 12px; margin-top: 16px; }
    pre { background: rgba(128, 128, 128, 0.15); padding: 12px; border-radius: 8px; white-space: pre-wrap; }
    #log { max-height: 240px; overflow: auto; }
  </style>
</head>
<body>
  <header>
    <h1>Vector's Might Blink Monitor</h1>
    <button id="copy">Copy</button>
    <span class="status" id="copy-status"></span>
  </header>

  <div class="row">
    <div>Search</div>
    <pre id="search"></pre>
  </div>
  <div class="row">
    <div>Fragment</div>
    <pre id="fragment"></pre>
  </div>
  <div class="row">
    <div>Event</div>
    <pre id="event"></pre>
  </div>
  <div class="row">
    <div>Stream</div>
    <pre id="log"></pre>
  </div>

  <script>
    const searchEl = document.getElementById('search');
    const fragmentEl = document.getElementById('fragment');
    const eventEl = document.getElementById('event');
    const logEl = document.getElementById('log');
    const copyBtn = document.getElementById('copy');
    const copyStatus = document.getElementById('copy-status');

    let lastPayload = null;
    const logLines = [];

    function renderLog() {
      logEl.textContent = logLines.join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setCopyStatus(message) {
      copyStatus.textContent = message;
      if (!message) return;
      setTimeout(() => { copyStatus.textContent = ''; }, 1500);
    }

    const source = new EventSource('/events');
    source.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        lastPayload = data;
        searchEl.textContent = data.search || '';
        fragmentEl.textContent = data.fragment || '';
        eventEl.textContent = JSON.stringify(data, null, 2);
        logLines.push(`[${new Date().toLocaleTimeString()}] chunk ${data.chunk} level ${data.level}`);
        if (logLines.length > 25) logLines.shift();
        renderLog();
      } catch (err) {
        console.error(err);
      }
    };

    copyBtn.addEventListener('click', async () => {
      if (!lastPayload) {
        setCopyStatus('no event yet');
        return;
      }
      const payload = `search: ${lastPayload.search || ''}\nchunk: ${lastPayload.chunk}\nfragment: ${lastPayload.fragment || ''}`;
      try {
        await navigator.clipboard.writeText(payload);
        setCopyStatus('copied');
      } catch (err) {
        setCopyStatus('copy failed');
      }
    });
  </script>
</body>
</html>"#;
const SIMILARITY_WEIGHT_THRESHOLD: f32 = 0.4;
const SIMILARITY_GROUP_THRESHOLD: f32 = 0.78;
const MAX_VECTOR_VIEW_POINTS: usize = 7;
const MAX_SIMILAR_VECTOR_POINTS: usize = 5;
const MAX_AUTO_RELATED_POINTS: usize = 20;
const MAX_QUERY_RESULTS: usize = 10;
const INVERT_NUM_STEPS: u32 = 20;
const INVERSION_CACHE_PATH: &str = "cache/inversions.json";
const EMBEDDING_CACHE_PATH: &str = "cache/embeddings.json";
const MIN_INVERSION_SCORE: f32 = 0.72;
const SEARCH_BOX_WIDTH: usize = 31;
const FILTERED_TEXT_LIMIT: usize = 14;
const SCANNER_ZOOM_MIN: f32 = 0.4;
const SCANNER_ZOOM_MAX: f32 = 10.0;
const SCANNER_ZOOM_STEP: f32 = 0.1;
const EMBED_LOAD_TIMEOUT_SECS: u64 = 120;
const EMBED_LOAD_GRACE_SECS: u64 = 240;
const INVERT_LOAD_TIMEOUT_SECS: u64 = 120;
const INVERT_LOAD_GRACE_SECS: u64 = 240;
const MAX_EMBED_RESTARTS: u8 = 2;
const MAX_INVERT_RESTARTS: u8 = 2;
const GPU_POLL_INTERVAL_SECS: u64 = 1;
const FUZZY_MATCH_THRESHOLD: f32 = 0.8;
const MIN_PHRASE_TOKENS: usize = 2;
const MAX_PHRASE_TOKENS: usize = 6;
const STOP_WORDS: &[&str] = &[
    "the", "and", "a", "an", "or", "but", "if", "in", "on", "of", "for", "to", "from",
    "by", "with", "is", "are", "was", "were", "be", "been", "being", "as", "at",
    "it", "this", "that", "these", "those", "we", "you", "i", "he", "she", "they",
    "them", "his", "her", "their", "our", "your", "its", "not", "no", "yes", "do",
    "does", "did", "done", "so", "than", "then", "too", "very", "can", "could",
    "should", "would", "will", "just", "into", "over", "under", "up", "down", "out",
    "about", "what", "http",
];

#[derive(Clone)]
struct ChunkInfo {
    index: usize,
    start: usize,
    end: usize,
    preview: String,
    text: String,
}

#[derive(Clone)]
struct VectorPoint {
    index: usize,
    coords: [f32; 3],
}

#[derive(Clone, Copy)]
struct VectorCell {
    ch: char,
    style: Option<Style>,
}

#[derive(Clone, Copy)]
struct GpuStatus {
    util: u8,
    mem_used: u32,
    mem_total: u32,
}

#[derive(Clone, Copy)]
struct FuzzyMatch {
    start: usize,
    end: usize,
    score: f32,
}

#[derive(Clone, Deserialize, Serialize)]
struct CachedInversion {
    text: String,
    score: f32,
}

#[derive(Deserialize, Serialize)]
struct EmbeddingCacheFile {
    window_tokens: usize,
    entries: HashMap<String, Vec<f32>>,
}

#[derive(Clone)]
struct TermCount {
    term: String,
    count: usize,
}

struct TermStats {
    count: usize,
    doc_freq: usize,
    max_chunk: usize,
}

struct ChunkSeed {
    start: usize,
    end: usize,
    text: String,
}

struct SeedAnalysis {
    chunks: Vec<ChunkInfo>,
}

#[derive(Clone, Copy, PartialEq)]
enum FocusArea {
    Search,
    SourceText,
    ChunkList,
    TermTicker,
}

#[derive(Clone, Copy, PartialEq)]
enum SimilarityMode {
    None,
    Browse,
    Groups,
    Query,
}

#[derive(Clone, Copy, PartialEq)]
enum SourceViewMode {
    Overview,
    Sorted,
}

struct App {
    input_path: Option<PathBuf>,
    text: String,
    chunks: Vec<ChunkInfo>,
    selected: usize,
    list_state: ListState,
    window_tokens: usize,
    status: String,
    score_config: ScoreConfig,
    processing: bool,
    auto_embed_pending: bool,
    processed_count: usize,
    active_index: Option<usize>,
    embeddings: Vec<Option<Vec<f32>>>,
    pending_embeddings: VecDeque<(usize, Vec<f32>)>,
    logs: VecDeque<String>,
    embed_started_at: Option<Instant>,
    last_embed_log: Option<Instant>,
    embed_done_pending: bool,
    last_embed_tick: Instant,
    source_scroll: usize,
    source_scroll_max: usize,
    source_page_size: usize,
    list_page_size: usize,
    vector_points: Vec<VectorPoint>,
    vector_point_cursor: usize,
    projection_axes: Option<[Vec<f32>; 3]>,
    projection_dirty: bool,
    orbit_angle: f32,
    orbit_speed: f32,
    time_enabled: bool,
    scanner_zoom: f32,
    gpu_status: Option<GpuStatus>,
    gpu_last_sample: Option<Instant>,
    gpu_available: Option<bool>,
    event_hub: Option<Arc<EventHub>>,
    blink_last: Vec<bool>,
    embed_ready: bool,
    embed_loading: bool,
    embed_last_event: Option<Instant>,
    embed_restarts: u8,
    invert_ready: bool,
    invert_loading: bool,
    invert_last_event: Option<Instant>,
    invert_restarts: u8,
    auto_related_scores: Vec<f32>,
    auto_related_counts: Vec<usize>,
    auto_related_indices: Vec<usize>,
    similarity_scores: Vec<f32>,
    similarity_mode: SimilarityMode,
    group_ids: Vec<Option<usize>>,
    show_markers: bool,
    show_similarity_menu: bool,
    similarity_menu_index: usize,
    search_query: String,
    search_query_snapshot: Option<String>,
    search_matches: Vec<(usize, usize)>,
    search_embedding: Option<Vec<f32>>,
    search_filter: Option<String>,
    search_result_indices: Vec<usize>,
    browse_result_indices: Vec<usize>,
    browse_anchor: Option<usize>,
    search_embed_inflight: bool,
    pending_window_delta: Option<i32>,
    show_extract_view: bool,
    extract_chunk_index: Option<usize>,
    extract_scroll: usize,
    extract_text: Option<String>,
    extract_score: Option<f32>,
    extract_status: String,
    extract_inflight: bool,
    extract_pending: Option<usize>,
    invert_queue: VecDeque<usize>,
    invert_inflight: Option<usize>,
    invert_done: Vec<bool>,
    invert_texts: Vec<Option<String>>,
    invert_scores: Vec<Option<f32>>,
    invert_queued: Vec<bool>,
    invert_all_active: bool,
    invert_total: usize,
    invert_completed: usize,
    invert_matches: Vec<Vec<FuzzyMatch>>,
    invert_prune_marked: Vec<bool>,
    invert_prune_boundary: Vec<Option<usize>>,
    invert_cache: HashMap<String, CachedInversion>,
    embedding_cache: HashMap<String, Vec<f32>>,
    embedding_cache_dirty: bool,
    corpus_words: Vec<String>,
    corpus_index: HashMap<char, Vec<usize>>,
    corpus_set: HashSet<String>,
    term_ticker: Vec<TermCount>,
    term_ticker_index: usize,
    term_ticker_scroll: usize,
    selected_terms: Vec<String>,
    term_scores: Vec<f32>,
    source_view: SourceViewMode,
    focus_area: FocusArea,
    cursor_blink_on: bool,
    cursor_blink_last: Instant,
}

#[derive(Clone)]
struct ScoreConfig {
    python_path: Option<PathBuf>,
    embed_script_path: PathBuf,
    model_server_path: PathBuf,
    model: String,
    max_length: u32,
    embed_batch_size: usize,
}

impl ScoreConfig {
    fn default_with_paths() -> Self {
        let embed_script_path = PathBuf::from("scripts/vec2text_embed.py");
        let model_server_path = PathBuf::from("scripts/model_server.py");
        let python_path = std::env::var("VECTOR_SMITE_PY")
            .ok()
            .map(PathBuf::from);
        let embed_batch_size = std::env::var("VECTORS_MIGHT_BATCH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_EMBED_BATCH_SIZE);
        Self {
            python_path,
            embed_script_path,
            model_server_path,
            model: "gtr-base".to_string(),
            max_length: DEFAULT_WINDOW_TOKENS as u32,
            embed_batch_size,
        }
    }

    fn is_embed_enabled(&self) -> bool {
        self.embed_script_path.exists()
    }

    fn is_invert_enabled(&self) -> bool {
        self.model_server_path.exists()
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    max_length: u32,
    batch_size: usize,
    texts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum EmbedEvent {
    Embedding { index: usize, embedding: Vec<f32> },
    Log { message: String },
    Done,
    Error { message: String },
}

struct ModelServerProcess {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<ModelServerEvent>,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum ModelServerEvent {
    Ready,
    Log { message: String },
    Result {
        command: String,
        inversions: Option<Vec<String>>,
        scores: Option<Vec<f32>>,
        id: Option<u64>,
    },
    Error { message: String, id: Option<u64> },
    Shutdown { id: Option<u64> },
}

impl App {
    fn new(input_path: Option<PathBuf>, score_config: ScoreConfig) -> Result<Self> {
        let mut app = Self {
            input_path,
            text: String::new(),
            chunks: Vec::new(),
            selected: 0,
            list_state: ListState::default(),
            window_tokens: score_config.max_length as usize,
            status: String::new(),
            score_config,
            processing: false,
            auto_embed_pending: false,
            processed_count: 0,
            active_index: None,
            embeddings: Vec::new(),
            pending_embeddings: VecDeque::new(),
            logs: VecDeque::new(),
            embed_started_at: None,
            last_embed_log: None,
            embed_done_pending: false,
            last_embed_tick: Instant::now(),
            source_scroll: 0,
            source_scroll_max: 0,
            source_page_size: 1,
            list_page_size: 1,
            vector_points: Vec::new(),
            vector_point_cursor: 0,
            projection_axes: None,
            projection_dirty: true,
            orbit_angle: 0.0,
            orbit_speed: 0.03,
            time_enabled: true,
            scanner_zoom: 1.0,
            gpu_status: None,
            gpu_last_sample: None,
            gpu_available: None,
            event_hub: None,
            blink_last: Vec::new(),
            embed_ready: false,
            embed_loading: false,
            embed_last_event: None,
            embed_restarts: 0,
            invert_ready: false,
            invert_loading: false,
            invert_last_event: None,
            invert_restarts: 0,
            auto_related_scores: Vec::new(),
            auto_related_counts: Vec::new(),
            auto_related_indices: Vec::new(),
            similarity_scores: Vec::new(),
            similarity_mode: SimilarityMode::None,
            group_ids: Vec::new(),
            show_markers: true,
            show_similarity_menu: false,
            similarity_menu_index: 0,
            search_query: String::new(),
            search_query_snapshot: None,
            search_matches: Vec::new(),
            search_embedding: None,
            search_filter: None,
            search_result_indices: Vec::new(),
            browse_result_indices: Vec::new(),
            browse_anchor: None,
            search_embed_inflight: false,
            pending_window_delta: None,
            show_extract_view: false,
            extract_chunk_index: None,
            extract_scroll: 0,
            extract_text: None,
            extract_score: None,
            extract_status: String::new(),
            extract_inflight: false,
            extract_pending: None,
            invert_queue: VecDeque::new(),
            invert_inflight: None,
            invert_done: Vec::new(),
            invert_texts: Vec::new(),
            invert_scores: Vec::new(),
            invert_queued: Vec::new(),
            invert_all_active: false,
            invert_total: 0,
            invert_completed: 0,
            invert_matches: Vec::new(),
            invert_prune_marked: Vec::new(),
            invert_prune_boundary: Vec::new(),
            invert_cache: HashMap::new(),
            embedding_cache: HashMap::new(),
            embedding_cache_dirty: false,
            corpus_words: Vec::new(),
            corpus_index: HashMap::new(),
            corpus_set: HashSet::new(),
            term_ticker: Vec::new(),
            term_ticker_index: 0,
            term_ticker_scroll: 0,
            selected_terms: Vec::new(),
            term_scores: Vec::new(),
            source_view: SourceViewMode::Overview,
            focus_area: FocusArea::ChunkList,
            cursor_blink_on: true,
            cursor_blink_last: Instant::now(),
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
        self.auto_embed_pending = false;
        self.processed_count = 0;
        self.active_index = None;
        self.embeddings.clear();
        self.pending_embeddings.clear();
        self.logs.clear();
        self.embed_started_at = None;
        self.last_embed_log = None;
        self.embed_done_pending = false;
        self.last_embed_tick = Instant::now();
        self.source_scroll = 0;
        self.source_scroll_max = 0;
        self.source_page_size = 1;
        self.list_page_size = 1;
        self.vector_points.clear();
        self.vector_point_cursor = 0;
        self.projection_axes = None;
        self.projection_dirty = true;
        self.embed_ready = false;
        self.embed_loading = false;
        self.embed_last_event = None;
        self.embed_restarts = 0;
        self.invert_ready = false;
        self.invert_loading = false;
        self.invert_last_event = None;
        self.invert_restarts = 0;
        self.auto_related_scores.clear();
        self.auto_related_counts.clear();
        self.auto_related_indices.clear();
        self.similarity_scores.clear();
        self.similarity_mode = SimilarityMode::None;
        self.group_ids.clear();
        self.show_markers = true;
        self.show_similarity_menu = false;
        self.similarity_menu_index = 0;
        self.search_query.clear();
        self.search_query_snapshot = None;
        self.search_matches.clear();
        self.search_embedding = None;
        self.search_filter = None;
        self.search_result_indices.clear();
        self.browse_result_indices.clear();
        self.browse_anchor = None;
        self.search_embed_inflight = false;
        self.pending_window_delta = None;
        self.show_extract_view = false;
        self.extract_chunk_index = None;
        self.extract_scroll = 0;
        self.extract_text = None;
        self.extract_score = None;
        self.extract_status.clear();
        self.extract_inflight = false;
        self.extract_pending = None;
        self.invert_queue.clear();
        self.invert_inflight = None;
        self.invert_done.clear();
        self.invert_texts.clear();
        self.invert_scores.clear();
        self.invert_queued.clear();
        self.invert_all_active = false;
        self.invert_total = 0;
        self.invert_completed = 0;
        self.invert_matches.clear();
        self.invert_prune_marked.clear();
        self.invert_prune_boundary.clear();
        self.invert_cache = load_inversion_cache();
        self.embedding_cache = load_embedding_cache(self.window_tokens);
        self.embedding_cache_dirty = false;
        self.corpus_words.clear();
        self.corpus_index.clear();
        self.corpus_set.clear();
        self.term_ticker.clear();
        self.term_ticker_index = 0;
        self.term_ticker_scroll = 0;
        self.selected_terms.clear();
        self.term_scores.clear();
        self.source_view = SourceViewMode::Overview;
        self.cursor_blink_on = true;
        self.cursor_blink_last = Instant::now();

        let seed = seed_chunks_from_tokens(&self.text, self.window_tokens);
        self.chunks = seed.chunks;
        self.blink_last = vec![false; self.chunks.len()];
        self.term_ticker = build_term_ticker(&self.invert_texts, &self.invert_scores, &self.corpus_set);
        self.term_ticker_index = 0;
        self.term_ticker_scroll = 0;
        self.selected_terms.clear();
        self.term_scores = vec![0.0; self.chunks.len()];
        self.embeddings = vec![None; self.chunks.len()];
        self.auto_related_scores = vec![0.0; self.chunks.len()];
        self.auto_related_counts = vec![0; self.chunks.len()];
        self.auto_related_indices.clear();
        self.similarity_scores = vec![0.0; self.chunks.len()];
        self.group_ids = vec![None; self.chunks.len()];
        self.invert_done = vec![false; self.chunks.len()];
        self.invert_texts = vec![None; self.chunks.len()];
        self.invert_scores = vec![None; self.chunks.len()];
        self.invert_queued = vec![false; self.chunks.len()];
        self.invert_matches = vec![Vec::new(); self.chunks.len()];
        self.invert_prune_marked = vec![false; self.chunks.len()];
        self.invert_prune_boundary = vec![None; self.chunks.len()];
        self.invert_total = self.chunks.len();
        self.invert_completed = 0;
        let (corpus_words, corpus_index, corpus_set) = build_corpus(&self.text);
        self.corpus_words = corpus_words;
        self.corpus_index = corpus_index;
        self.corpus_set = corpus_set;
        self.term_ticker = build_term_ticker(&self.invert_texts, &self.invert_scores, &self.corpus_set);
        self.selected = 0.min(self.chunks.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
        let cached = self.apply_embedding_cache();
        self.processed_count = cached;
        let missing = self.chunks.len().saturating_sub(cached);
        if cached == self.chunks.len() && self.similarity_mode == SimilarityMode::None {
            self.compute_similarity_groups();
        }
        if missing == 0 {
            self.status = format!(
                "loaded {} chars | seed {} chunks | fixed {} | cached {}/{} embeddings",
                self.text.chars().count(),
                self.chunks.len(),
                self.window_tokens,
                cached,
                self.chunks.len()
            );
            self.push_log(&format!("embedding cache hit {}/{}", cached, self.chunks.len()));
            self.auto_embed_pending = false;
        } else {
            self.status = format!(
                "loaded {} chars | seed {} chunks | fixed {} | cached {}/{} | auto-embedding...",
                self.text.chars().count(),
                self.chunks.len(),
                self.window_tokens,
                cached,
                self.chunks.len()
            );
            self.push_log(&format!(
                "embedding cache hit {}/{}",
                cached,
                self.chunks.len()
            ));
            self.push_log("seeded chunks, auto-embedding...");
            self.auto_embed_pending = true;
        }
        Ok(())
    }

    fn update_search(&mut self) {
        self.search_matches.clear();
        if self.search_query.len() < 2 {
            if !self.search_query.is_empty() {
                self.status = "type more to search...".to_string();
            }
            return;
        }
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
            let resume_invert = self.invert_all_active
                || self.invert_done.iter().any(|done| *done);
            self.window_tokens = next;
            self.score_config.max_length = next as u32;
            let seed = seed_chunks_from_tokens(&self.text, self.window_tokens);
            self.chunks = seed.chunks;
            self.blink_last = vec![false; self.chunks.len()];
            self.term_ticker = build_term_ticker(&self.invert_texts, &self.invert_scores, &self.corpus_set);
            self.term_ticker_index = 0;
            self.term_ticker_scroll = 0;
            self.selected_terms.clear();
            self.term_scores = vec![0.0; self.chunks.len()];
            self.embeddings = vec![None; self.chunks.len()];
            self.similarity_scores = vec![0.0; self.chunks.len()];
            self.auto_related_scores = vec![0.0; self.chunks.len()];
            self.auto_related_counts = vec![0; self.chunks.len()];
            self.auto_related_indices.clear();
            self.vector_points.clear();
            self.vector_point_cursor = 0;
            self.projection_axes = None;
            self.projection_dirty = true;
            self.embed_ready = false;
            self.embed_loading = false;
            self.embed_last_event = None;
            self.embed_restarts = 0;
            self.invert_ready = false;
            self.invert_loading = false;
            self.invert_last_event = None;
            self.invert_restarts = 0;
            self.similarity_mode = SimilarityMode::None;
            self.group_ids = vec![None; self.chunks.len()];
            self.show_similarity_menu = false;
            self.similarity_menu_index = 0;
            self.pending_embeddings.clear();
            self.search_embedding = None;
            self.search_filter = None;
            self.search_result_indices.clear();
            self.browse_result_indices.clear();
            self.search_query_snapshot = None;
            self.search_embed_inflight = false;
            self.pending_window_delta = None;
            self.show_extract_view = false;
            self.extract_chunk_index = None;
            self.extract_scroll = 0;
            self.extract_text = None;
            self.extract_score = None;
            self.extract_status.clear();
            self.extract_inflight = false;
            self.extract_pending = None;
            self.invert_queue.clear();
            self.invert_inflight = None;
            self.invert_done = vec![false; self.chunks.len()];
            self.invert_texts = vec![None; self.chunks.len()];
            self.invert_scores = vec![None; self.chunks.len()];
            self.invert_queued = vec![false; self.chunks.len()];
            self.invert_matches = vec![Vec::new(); self.chunks.len()];
            self.invert_prune_marked = vec![false; self.chunks.len()];
            self.invert_prune_boundary = vec![None; self.chunks.len()];
            self.invert_all_active = resume_invert;
            self.invert_total = self.chunks.len();
            self.invert_completed = 0;
            self.embedding_cache = load_embedding_cache(self.window_tokens);
            self.embedding_cache_dirty = false;
            let (corpus_words, corpus_index, corpus_set) = build_corpus(&self.text);
            self.corpus_words = corpus_words;
            self.corpus_index = corpus_index;
            self.corpus_set = corpus_set;
            self.term_ticker = build_term_ticker(&self.invert_texts, &self.invert_scores, &self.corpus_set);
            self.selected = 0.min(self.chunks.len().saturating_sub(1));
            self.list_state.select(Some(self.selected));
            let cached = self.apply_embedding_cache();
            self.processed_count = cached;
            let missing = self.chunks.len().saturating_sub(cached);
            if cached == self.chunks.len() && self.similarity_mode == SimilarityMode::None {
                self.compute_similarity_groups();
            }
            if missing == 0 {
                self.status = format!(
                    "fixed {} | seed {} chunks | cached {}/{} embeddings",
                    self.window_tokens,
                    self.chunks.len(),
                    cached,
                    self.chunks.len()
                );
                self.push_log(&format!("embedding cache hit {}/{}", cached, self.chunks.len()));
                self.auto_embed_pending = false;
            } else {
                self.status = format!(
                    "fixed {} | seed {} chunks | cached {}/{} | auto-embedding...",
                    self.window_tokens,
                    self.chunks.len(),
                    cached,
                    self.chunks.len()
                );
                self.push_log(&format!("embedding cache hit {}/{}", cached, self.chunks.len()));
                self.push_log("seeded chunks, auto-embedding...");
                self.auto_embed_pending = true;
            }
        }
    }

    fn queue_window_adjust(&mut self, delta: i32) {
        let next_delta = self.pending_window_delta.unwrap_or(0) + delta;
        if next_delta == 0 {
            self.pending_window_delta = None;
            self.status = format!("fixed {}", self.window_tokens);
            return;
        }
        self.pending_window_delta = Some(next_delta);
        let target = (self.window_tokens as i32 + next_delta).max(8);
        self.status = format!("fixed {} -> {} (queued)", self.window_tokens, target);
    }

    fn select_prev(&mut self) {
        let indices = current_list_indices(self);
        if indices.is_empty() {
            return;
        }
        let pos = indices.iter().position(|idx| *idx == self.selected).unwrap_or(0);
        let new_pos = pos.saturating_sub(1);
        self.selected = indices[new_pos];
        self.list_state.select(Some(self.selected));
        if self.similarity_mode == SimilarityMode::Browse {
            self.compute_similarity_scores();
        }
    }

    fn select_next(&mut self) {
        let indices = current_list_indices(self);
        if indices.is_empty() {
            return;
        }
        let pos = indices.iter().position(|idx| *idx == self.selected).unwrap_or(0);
        let new_pos = if pos + 1 < indices.len() { pos + 1 } else { pos };
        self.selected = indices[new_pos];
        self.list_state.select(Some(self.selected));
        if self.similarity_mode == SimilarityMode::Browse {
            self.compute_similarity_scores();
        }
    }

    fn page_chunk_list(&mut self, forward: bool) {
        let indices = current_list_indices(self);
        if indices.is_empty() {
            return;
        }
        let pos = indices.iter().position(|idx| *idx == self.selected).unwrap_or(0);
        let page = self.list_page_size.max(1);
        let new_pos = if forward {
            (pos + page).min(indices.len().saturating_sub(1))
        } else {
            pos.saturating_sub(page)
        };
        self.selected = indices[new_pos];
        self.list_state.select(Some(self.selected));
        if self.similarity_mode == SimilarityMode::Browse {
            self.compute_similarity_scores();
        }
    }

    fn select_term_prev(&mut self) {
        if self.term_ticker.is_empty() {
            return;
        }
        if self.term_ticker_index == 0 {
            self.term_ticker_index = self.term_ticker.len().saturating_sub(1);
        } else {
            self.term_ticker_index = self.term_ticker_index.saturating_sub(1);
        }
        self.refresh_term_scores();
    }

    fn select_term_next(&mut self) {
        if self.term_ticker.is_empty() {
            return;
        }
        if self.term_ticker_index + 1 >= self.term_ticker.len() {
            self.term_ticker_index = 0;
        } else {
            self.term_ticker_index += 1;
        }
        self.refresh_term_scores();
    }

    fn active_term(&self) -> Option<&str> {
        if !self.selected_terms.is_empty() {
            return None;
        }
        let term = self.term_ticker.get(self.term_ticker_index)?.term.as_str();
        if term == "All terms" {
            None
        } else {
            Some(term)
        }
    }

    fn term_bias_active(&self) -> bool {
        !self.selected_terms.is_empty() || self.active_term().is_some()
    }

    fn reset_term_selection(&mut self) {
        self.selected_terms.clear();
        self.term_ticker_index = 0;
        self.term_ticker_scroll = 0;
        self.refresh_term_scores();
    }

    fn toggle_term_selection(&mut self) {
        if self.term_ticker.is_empty() {
            return;
        }
        let term = self.term_ticker[self.term_ticker_index].term.clone();
        if term == "All terms" {
            self.reset_term_selection();
            self.status = "terms cleared".to_string();
            return;
        }
        if let Some(pos) = self.selected_terms.iter().position(|item| item == &term) {
            self.selected_terms.remove(pos);
            if self.selected_terms.is_empty() {
                self.reset_term_selection();
                self.status = "terms cleared".to_string();
                return;
            }
        } else {
            self.selected_terms.push(term);
        }
        self.refresh_term_scores();
    }

    fn selected_terms_label(&self) -> Option<String> {
        if self.selected_terms.is_empty() {
            return None;
        }
        Some(format!("terms: {}", self.selected_terms.join(", ")))
    }

    fn refresh_term_scores(&mut self) {
        self.term_scores = vec![0.0; self.chunks.len()];
        self.projection_dirty = true;
        if !self.selected_terms.is_empty() {
            for term in &self.selected_terms {
                let term_tokens: Vec<String> = term
                    .split_whitespace()
                    .map(normalize_token)
                    .filter(|token| !token.is_empty())
                    .collect();
                if term_tokens.is_empty() {
                    continue;
                }
                for (idx, text) in self.invert_texts.iter().enumerate() {
                    let score = self
                        .invert_scores
                        .get(idx)
                        .and_then(|value| *value)
                        .unwrap_or(0.0);
                    if score < MIN_INVERSION_SCORE {
                        continue;
                    }
                    if let Some(text) = text.as_deref() {
                        let score = score_term_in_text(&term_tokens, text);
                        if let Some(slot) = self.term_scores.get_mut(idx) {
                            *slot += score;
                        }
                    }
                }
            }
            self.align_orbit_to_term();
            return;
        }
        let Some(term) = self.active_term() else {
            return;
        };
        let term_tokens: Vec<String> = term
            .split_whitespace()
            .map(normalize_token)
            .filter(|token| !token.is_empty())
            .collect();
        if term_tokens.is_empty() {
            return;
        }
        for (idx, text) in self.invert_texts.iter().enumerate() {
            let score = self
                .invert_scores
                .get(idx)
                .and_then(|value| *value)
                .unwrap_or(0.0);
            if score < MIN_INVERSION_SCORE {
                continue;
            }
            if let Some(text) = text.as_deref() {
                let score = score_term_in_text(&term_tokens, text);
                if let Some(slot) = self.term_scores.get_mut(idx) {
                    *slot = score;
                }
            }
        }
        self.align_orbit_to_term();
    }

    fn align_orbit_to_term(&mut self) {
        let mut scored: Vec<(usize, f32)> = self
            .term_scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| *score > 0.0)
            .collect();
        if scored.is_empty() {
            return;
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let top_indices: Vec<usize> = scored
            .iter()
            .take(MAX_VECTOR_VIEW_POINTS)
            .map(|(idx, _)| *idx)
            .collect();

        let mut emb_samples: Vec<Vec<f32>> = Vec::new();
        for idx in &top_indices {
            if let Some(emb) = self.embeddings.get(*idx).and_then(|emb| emb.as_ref()) {
                emb_samples.push(emb.clone());
            }
        }
        if emb_samples.is_empty() {
            return;
        }
        let mut points: Vec<[f32; 3]> = Vec::new();
        for emb in &emb_samples {
            points.push(embedding_to_point(self, emb));
        }

        let mut center = [0.0f32; 3];
        for point in &points {
            center[0] += point[0];
            center[1] += point[1];
            center[2] += point[2];
        }
        let denom = points.len() as f32;
        center[0] /= denom;
        center[1] /= denom;
        center[2] /= denom;

        let target_idx = scored[0].0;
        let Some(target_emb) = self.embeddings.get(target_idx).and_then(|emb| emb.as_ref()) else {
            return;
        };
        let target_emb = target_emb.clone();
        let target_point = embedding_to_point(self, &target_emb);
        let rel_x = target_point[0] - center[0];
        let rel_z = target_point[2] - center[2];
        let angle = rel_x.atan2(rel_z);
        self.orbit_angle = -angle;
    }

    fn refresh_term_ticker(&mut self) {
        let previous = self
            .term_ticker
            .get(self.term_ticker_index)
            .map(|term| term.term.clone());
        self.term_ticker =
            build_term_ticker(&self.invert_texts, &self.invert_scores, &self.corpus_set);
        if let Some(prev) = previous {
            if let Some(pos) = self.term_ticker.iter().position(|term| term.term == prev) {
                self.term_ticker_index = pos;
            } else {
                self.term_ticker_index = 0;
            }
        } else {
            self.term_ticker_index = 0;
        }
        self.term_ticker_scroll = 0;
        self.refresh_term_scores();
    }

    fn adjust_scanner_zoom(&mut self, delta: f32) {
        let next = (self.scanner_zoom + delta).clamp(SCANNER_ZOOM_MIN, SCANNER_ZOOM_MAX);
        if (next - self.scanner_zoom).abs() > f32::EPSILON {
            self.scanner_zoom = next;
            self.status = format!("scanner zoom {:.1}x", self.scanner_zoom);
        }
    }

    fn note_embed_event(&mut self) {
        self.embed_last_event = Some(Instant::now());
    }

    fn mark_embed_ready(&mut self) {
        self.embed_ready = true;
        self.embed_loading = false;
        self.embed_last_event = Some(Instant::now());
    }

    fn embedder_should_restart(&self) -> bool {
        if self.embed_ready || self.embed_restarts >= MAX_EMBED_RESTARTS {
            return false;
        }
        let Some(start) = self.embed_started_at else {
            return false;
        };
        let timeout = if self.embed_loading {
            EMBED_LOAD_GRACE_SECS
        } else {
            EMBED_LOAD_TIMEOUT_SECS
        };
        if start.elapsed() < Duration::from_secs(timeout) {
            return false;
        }
        if let Some(last) = self.embed_last_event {
            if last.elapsed() < Duration::from_secs(timeout) {
                return false;
            }
        }
        true
    }

    fn note_invert_event(&mut self) {
        self.invert_last_event = Some(Instant::now());
    }

    fn mark_invert_ready(&mut self) {
        self.invert_ready = true;
        self.invert_loading = false;
        self.invert_last_event = Some(Instant::now());
    }

    fn invert_work_pending(&self) -> bool {
        self.invert_inflight.is_some()
            || !self.invert_queue.is_empty()
            || self.extract_inflight
            || self.extract_pending.is_some()
    }

    fn invert_should_restart(&self) -> bool {
        if self.invert_ready || self.invert_restarts >= MAX_INVERT_RESTARTS {
            return false;
        }
        if !self.invert_work_pending() {
            return false;
        }
        let Some(start) = self.invert_last_event else {
            return false;
        };
        let timeout = if self.invert_loading {
            INVERT_LOAD_GRACE_SECS
        } else {
            INVERT_LOAD_TIMEOUT_SECS
        };
        start.elapsed() >= Duration::from_secs(timeout)
    }

    fn push_log(&mut self, message: &str) {
        if message.is_empty() {
            return;
        }
        if self.logs.len() >= MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(message.to_string());
    }

    fn compute_similarity_scores(&mut self) {
        self.similarity_scores.fill(0.0);
        let Some(selected_emb) = self
            .embeddings
            .get(self.selected)
            .and_then(|emb| emb.as_ref())
        else {
            self.push_log("no embedding for selected chunk");
            return;
        };

        for (idx, emb) in self.embeddings.iter().enumerate() {
            if let Some(vec) = emb.as_ref() {
                let sim = cosine_similarity(selected_emb, vec);
                self.similarity_scores[idx] = sim.max(0.0);
            }
        }
    }

    fn clear_similarity(&mut self) {
        self.similarity_mode = SimilarityMode::None;
        self.similarity_scores.fill(0.0);
        self.group_ids.fill(None);
        self.search_embedding = None;
        self.search_filter = None;
        self.search_result_indices.clear();
        self.browse_result_indices.clear();
        self.browse_anchor = None;
        self.search_embed_inflight = false;
        self.pending_window_delta = None;
        self.show_extract_view = false;
        self.extract_chunk_index = None;
        self.extract_scroll = 0;
        self.extract_text = None;
        self.extract_score = None;
        self.extract_status.clear();
        self.extract_inflight = false;
        self.extract_pending = None;
        self.status = "similarity cleared".to_string();
    }

    fn start_browse_similarity(&mut self) {
        if self
            .embeddings
            .get(self.selected)
            .and_then(|emb| emb.as_ref())
            .is_none()
        {
            self.status = "no embedding for selected chunk".to_string();
            return;
        }
        self.browse_anchor = Some(self.selected);
        self.compute_similarity_scores();
        self.similarity_mode = SimilarityMode::Browse;
        self.rebuild_browse_results();
        self.status = format!("showing similarity for chunk {}", self.selected + 1);
    }

    fn compute_similarity_groups(&mut self) {
        if self.embeddings.iter().all(|emb| emb.is_none()) {
            self.status = "no embeddings available".to_string();
            return;
        }
        let mut group_ids = vec![None; self.chunks.len()];
        let mut group_count = 0usize;
        for i in 0..self.chunks.len() {
            if group_ids[i].is_some() {
                continue;
            }
            let Some(emb_i) = self.embeddings[i].as_ref() else {
                continue;
            };
            group_ids[i] = Some(group_count);
            for j in (i + 1)..self.chunks.len() {
                if group_ids[j].is_some() {
                    continue;
                }
                if let Some(emb_j) = self.embeddings[j].as_ref() {
                    let sim = cosine_similarity(emb_i, emb_j);
                    if sim >= SIMILARITY_GROUP_THRESHOLD {
                        group_ids[j] = Some(group_count);
                    }
                }
            }
            group_count += 1;
        }
        if group_count == 0 {
            self.status = "no similarity groups found".to_string();
            return;
        }
        self.group_ids = group_ids;
        self.similarity_scores.fill(0.0);
        self.similarity_mode = SimilarityMode::Groups;
        self.status = format!("computed {} similarity groups", group_count);
    }

    fn update_vector_point(&mut self, index: usize, embedding: &[f32]) {
        self.projection_dirty = true;
        let coords = embedding_to_point(self, embedding);
        if let Some(existing) = self.vector_points.iter_mut().find(|point| point.index == index) {
            existing.coords = coords;
            return;
        }

        if self.vector_points.len() < MAX_VECTOR_VIEW_POINTS {
            self.vector_points.push(VectorPoint { index, coords });
            return;
        }

        let slot = self.vector_point_cursor % MAX_VECTOR_VIEW_POINTS;
        if let Some(existing) = self.vector_points.get_mut(slot) {
            *existing = VectorPoint { index, coords };
        }
        self.vector_point_cursor = (self.vector_point_cursor + 1) % MAX_VECTOR_VIEW_POINTS;
    }

    fn update_auto_related(&mut self, new_index: usize) {
        let Some(new_emb) = self.embeddings.get(new_index).and_then(|emb| emb.as_ref()) else {
            return;
        };
        for (idx, emb) in self.embeddings.iter().enumerate() {
            if idx == new_index {
                continue;
            }
            let Some(other) = emb.as_ref() else {
                continue;
            };
            let sim = cosine_similarity(new_emb, other).max(0.0);
            if let Some(score) = self.auto_related_scores.get_mut(idx) {
                *score += sim;
            }
            if let Some(score) = self.auto_related_scores.get_mut(new_index) {
                *score += sim;
            }
            if let Some(count) = self.auto_related_counts.get_mut(idx) {
                *count += 1;
            }
            if let Some(count) = self.auto_related_counts.get_mut(new_index) {
                *count += 1;
            }
        }
        self.rebuild_auto_related_indices();
    }

    fn rebuild_auto_related_indices(&mut self) {
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| {
                emb.as_ref()?;
                let count = self.auto_related_counts.get(idx).copied().unwrap_or(0);
                let score = if count > 0 {
                    self.auto_related_scores.get(idx).copied().unwrap_or(0.0) / count as f32
                } else {
                    0.0
                };
                Some((idx, score))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(MAX_AUTO_RELATED_POINTS);
        self.auto_related_indices = scored.into_iter().map(|(idx, _)| idx).collect();
    }

    fn apply_search_embedding(&mut self, embedding: Vec<f32>) {
        self.search_embedding = Some(embedding.clone());
        self.similarity_scores.fill(0.0);

        for (idx, emb) in self.embeddings.iter().enumerate() {
            if let Some(vec) = emb.as_ref() {
                let sim = cosine_similarity(&embedding, vec).max(0.0);
                if let Some(score) = self.similarity_scores.get_mut(idx) {
                    *score = sim;
                }
            }
        }
        self.similarity_mode = SimilarityMode::Query;
        self.rebuild_search_results();
        if let Some(filter) = &self.search_filter {
            self.status = format!("filtering with \"{}\"", filter);
        }
    }

    fn rebuild_search_results(&mut self) {
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter_map(|(idx, emb)| {
                emb.as_ref()?;
                let score = self.similarity_scores.get(idx).copied().unwrap_or(0.0);
                Some((idx, score))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(MAX_QUERY_RESULTS);
        self.search_result_indices = scored.into_iter().map(|(idx, _)| idx).collect();
        if self.similarity_mode == SimilarityMode::Query && !self.search_result_indices.is_empty() {
            if !self.search_result_indices.contains(&self.selected) {
                self.selected = self.search_result_indices[0];
                self.list_state.select(Some(self.selected));
            }
        }
    }

    fn rebuild_browse_results(&mut self) {
        let mut indices: Vec<usize> = (0..self.chunks.len()).collect();
        indices.sort_by(|a, b| {
            let score_a = self.similarity_scores.get(*a).copied().unwrap_or(0.0);
            let score_b = self.similarity_scores.get(*b).copied().unwrap_or(0.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        self.browse_result_indices = indices;
        if self.similarity_mode == SimilarityMode::Browse && !self.browse_result_indices.is_empty() {
            if !self.browse_result_indices.contains(&self.selected) {
                self.selected = self.browse_result_indices[0];
                self.list_state.select(Some(self.selected));
            }
        }
    }

    fn queue_inversion(&mut self, index: usize) {
        if index >= self.chunks.len() {
            return;
        }
        if self
            .invert_done
            .get(index)
            .copied()
            .unwrap_or(false)
        {
            return;
        }
        if self
            .invert_queued
            .get(index)
            .copied()
            .unwrap_or(false)
        {
            return;
        }
        if self.invert_inflight == Some(index) {
            return;
        }
        if self
            .embeddings
            .get(index)
            .and_then(|emb| emb.as_ref())
            .is_none()
        {
            return;
        }
        if let Some(flag) = self.invert_queued.get_mut(index) {
            *flag = true;
        }
        self.invert_queue.push_back(index);
    }

    fn start_invert_all(&mut self) -> Result<(), String> {
        if !self.score_config.is_invert_enabled() {
            return Err("vec2text model server missing".to_string());
        }
        if self.chunks.is_empty() {
            return Err("no chunks to invert".to_string());
        }
        self.invert_all_active = true;
        self.invert_total = self.chunks.len();
        self.invert_completed = self.invert_done.iter().filter(|done| **done).count();
        self.invert_queue.clear();
        for flag in &mut self.invert_queued {
            *flag = false;
        }
        for idx in 0..self.chunks.len() {
            self.queue_inversion(idx);
        }
        if self.invert_queue.is_empty() && self.invert_inflight.is_none() {
            self.status = format!(
                "inversion up to date {}/{}",
                self.invert_completed, self.invert_total
            );
        } else {
            self.status = format!(
                "inverting {}/{}",
                self.invert_completed, self.invert_total
            );
        }
        Ok(())
    }

    fn open_extract_view(&mut self) -> Result<(), String> {
        if !self.score_config.is_invert_enabled() {
            return Err("vec2text model server missing".to_string());
        }
        let index = self.selected;
        self.extract_chunk_index = Some(index);
        self.extract_scroll = 0;
        self.show_extract_view = true;

        if let Some(text) = self
            .invert_texts
            .get(index)
            .and_then(|value| value.clone())
        {
            self.extract_text = Some(text.clone());
            self.extract_score = self.invert_scores.get(index).copied().flatten();
            self.extract_status = "inversion cached".to_string();
            self.extract_inflight = false;
            self.extract_pending = None;
            return Ok(());
        }

        if self.extract_inflight {
            return Err("vec2text inversion in progress".to_string());
        }
        self
            .embeddings
            .get(index)
            .and_then(|emb| emb.as_ref())
            .ok_or_else(|| "no embedding for selected chunk".to_string())?;
        self.extract_text = None;
        self.extract_score = None;
        self.extract_status = "queued for inversion".to_string();
        self.extract_pending = Some(index);
        Ok(())
    }

    fn apply_inversion_cache(&mut self, index: usize, embedding: &[f32]) {
        let hash = hash_embedding(embedding);
        let Some(cached) = self.invert_cache.get(&hash).cloned() else {
            return;
        };
        if let Some(slot) = self.invert_texts.get_mut(index) {
            *slot = Some(cached.text.clone());
        }
        if let Some(slot) = self.invert_scores.get_mut(index) {
            *slot = Some(cached.score);
        }
        if let Some(done) = self.invert_done.get_mut(index) {
            if !*done {
                *done = true;
                self.invert_completed += 1;
            }
        }
        self.update_inversion_matches(index, &cached.text);
    }

    fn apply_embedding_cache(&mut self) -> usize {
        if self.embedding_cache.is_empty() {
            return 0;
        }
        let mut cached = 0usize;
        for idx in 0..self.chunks.len() {
            let key = chunk_cache_key(&self.chunks[idx].text);
            let embedding = match self.embedding_cache.get(&key).cloned() {
                Some(value) => value,
                None => continue,
            };
            if self.embeddings.get(idx).and_then(|emb| emb.as_ref()).is_some() {
                continue;
            }
            if let Some(slot) = self.embeddings.get_mut(idx) {
                *slot = Some(embedding.clone());
            }
            self.update_vector_point(idx, &embedding);
            self.apply_inversion_cache(idx, &embedding);
            if self.invert_all_active {
                self.queue_inversion(idx);
            }
            self.update_auto_related(idx);
            cached += 1;
        }
        cached
    }

    fn update_inversion_matches(&mut self, index: usize, inversion_text: &str) {
        if index >= self.chunks.len() {
            return;
        }
        let score = self
            .invert_scores
            .get(index)
            .and_then(|value| *value)
            .unwrap_or(0.0);
        if score < MIN_INVERSION_SCORE {
            if let Some(slot) = self.invert_matches.get_mut(index) {
                slot.clear();
            }
            if let Some(slot) = self.invert_prune_boundary.get_mut(index) {
                *slot = None;
            }
            if let Some(mark) = self.invert_prune_marked.get_mut(index) {
                *mark = false;
            }
            self.refresh_term_ticker();
            return;
        }
        if inversion_text.trim().is_empty() {
            if let Some(slot) = self.invert_matches.get_mut(index) {
                slot.clear();
            }
            return;
        }
        let Some(chunk) = self.chunks.get(index) else {
            return;
        };
        let ranges = collect_token_ranges(&chunk.text);
        if ranges.is_empty() {
            if let Some(slot) = self.invert_matches.get_mut(index) {
                slot.clear();
            }
            return;
        }
        let mut chunk_tokens: HashSet<String> = HashSet::new();
        for (_, _, token) in &ranges {
            if !token.is_empty() {
                chunk_tokens.insert(token.clone());
            }
        }
        let prune_boundary = compute_prune_boundary(inversion_text, &chunk_tokens);
        let inversion_tokens: Vec<String> = inversion_text
            .split_whitespace()
            .map(normalize_token)
            .filter(|token| !token.is_empty())
            .collect();
        let mut phrase_set: HashSet<String> = HashSet::new();
        if inversion_tokens.len() >= MIN_PHRASE_TOKENS {
            for start_idx in 0..inversion_tokens.len() {
                for len in MIN_PHRASE_TOKENS..=MAX_PHRASE_TOKENS {
                    if start_idx + len > inversion_tokens.len() {
                        break;
                    }
                    let slice = &inversion_tokens[start_idx..start_idx + len];
                    if slice.iter().all(|token| is_stop_word(token)) {
                        continue;
                    }
                    phrase_set.insert(slice.join(" "));
                }
            }
        }

        let mut matches: Vec<FuzzyMatch> = Vec::new();
        if !phrase_set.is_empty() {
            let mut seq: Vec<&str> = Vec::with_capacity(ranges.len());
            for (_, _, token) in &ranges {
                seq.push(token.as_str());
            }
            for start_idx in 0..seq.len() {
                for len in MIN_PHRASE_TOKENS..=MAX_PHRASE_TOKENS {
                    if start_idx + len > seq.len() {
                        break;
                    }
                    let phrase = seq[start_idx..start_idx + len].join(" ");
                    if phrase_set.contains(&phrase) {
                        let start = ranges[start_idx].0;
                        let end = ranges[start_idx + len - 1].1;
                        let score = 1.0 + (len as f32 * 0.05);
                        matches.push(FuzzyMatch {
                            start: chunk.start + start,
                            end: chunk.start + end,
                            score,
                        });
                    }
                }
            }
        }

        for token in &inversion_tokens {
            if is_stop_word(token) {
                continue;
            }
            if !self.corpus_set.contains(token) {
                let pos = self.corpus_words.len();
                self.corpus_words.push(token.clone());
                self.corpus_set.insert(token.clone());
                if let Some(first) = token.chars().next() {
                    self.corpus_index.entry(first).or_default().push(pos);
                }
            }
            if !is_matchable_token(token) {
                continue;
            }
            let token_first = token.chars().next();
            for (start, end, chunk_token) in &ranges {
                if chunk_token.is_empty() {
                    continue;
                }
                if token_first != chunk_token.chars().next() {
                    continue;
                }
                let score = if chunk_token == token {
                    1.0
                } else {
                    levenshtein_similarity(token, chunk_token)
                };
                if score >= FUZZY_MATCH_THRESHOLD {
                    matches.push(FuzzyMatch {
                        start: chunk.start + *start,
                        end: chunk.start + *end,
                        score,
                    });
                }
            }
        }
        let merged = merge_fuzzy_matches(matches);
        if let Some(slot) = self.invert_matches.get_mut(index) {
            *slot = merged;
        }
        if let Some(slot) = self.invert_prune_boundary.get_mut(index) {
            *slot = prune_boundary;
        }
        if let Some(mark) = self.invert_prune_marked.get_mut(index) {
            *mark = false;
        }
        self.refresh_term_ticker();
    }

    fn prune_available(&self, index: usize) -> bool {
        let Some(text) = self
            .invert_texts
            .get(index)
            .and_then(|value| value.as_ref())
        else {
            return false;
        };
        let Some(boundary) = self
            .invert_prune_boundary
            .get(index)
            .and_then(|value| *value)
        else {
            return false;
        };
        let trimmed_len = text.trim_end().len();
        boundary < trimmed_len
    }

    fn toggle_prune_mark(&mut self) {
        let index = self.selected;
        if !self.prune_available(index) {
            self.status = "no prune target".to_string();
            return;
        }
        if let Some(mark) = self.invert_prune_marked.get_mut(index) {
            *mark = !*mark;
            self.status = if *mark {
                "prune marked (P to apply)".to_string()
            } else {
                "prune cleared".to_string()
            };
        }
    }

    fn apply_prune_marks(&mut self) {
        let mut pruned = 0usize;
        let mut cache_dirty = false;
        for idx in 0..self.chunks.len() {
            if !self
                .invert_prune_marked
                .get(idx)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some(text) = self
                .invert_texts
                .get(idx)
                .and_then(|value| value.as_ref())
                .cloned()
            else {
                continue;
            };
            let Some(boundary) = self
                .invert_prune_boundary
                .get(idx)
                .and_then(|value| *value)
            else {
                continue;
            };
            let trimmed = text[..boundary].trim_end().to_string();
            if trimmed.len() >= text.len() {
                if let Some(mark) = self.invert_prune_marked.get_mut(idx) {
                    *mark = false;
                }
                continue;
            }
            if let Some(slot) = self.invert_texts.get_mut(idx) {
                *slot = Some(trimmed.clone());
            }
            self.update_inversion_matches(idx, &trimmed);
            if let Some(mark) = self.invert_prune_marked.get_mut(idx) {
                *mark = false;
            }
            if let Some(embedding) = self
                .embeddings
                .get(idx)
                .and_then(|emb| emb.as_ref())
            {
                let score = self
                    .invert_scores
                    .get(idx)
                    .copied()
                    .flatten()
                    .unwrap_or(0.0);
                self.invert_cache.insert(
                    hash_embedding(embedding),
                    CachedInversion {
                        text: trimmed,
                        score,
                    },
                );
                cache_dirty = true;
            }
            pruned += 1;
        }
        if cache_dirty {
            if let Err(err) = save_inversion_cache(&self.invert_cache) {
                self.push_log(&format!("cache save failed: {}", err));
            }
        }
        if pruned > 0 {
            self.status = format!("pruned {} chunk{}", pruned, if pruned == 1 { "" } else { "s" });
        } else {
            self.status = "no prune marks".to_string();
        }
    }

    fn update_processing_status(&mut self) {
        if !self.processing {
            return;
        }
        if self.processed_count > 0 || !self.pending_embeddings.is_empty() {
            return;
        }
        let spinner = ['|', '/', '-', '\\'];
        let idx = (self.last_embed_tick.elapsed().as_millis() / 200) as usize % spinner.len();
        let window_status = if let Some(delta) = self.pending_window_delta {
            if delta != 0 {
                let target = (self.window_tokens as i32 + delta).max(8) as usize;
                format!("fixed {} -> {} (queued)", self.window_tokens, target)
            } else {
                format!("fixed {}", self.window_tokens)
            }
        } else {
            format!("fixed {}", self.window_tokens)
        };
        self.status = format!(
            "processing {} {}/{} | {}",
            spinner[idx],
            self.processed_count,
            self.chunks.len(),
            window_status
        );
    }

    fn update_gpu_status(&mut self) {
        if matches!(self.gpu_available, Some(false)) {
            return;
        }
        if let Some(last) = self.gpu_last_sample {
            if last.elapsed() < Duration::from_secs(GPU_POLL_INTERVAL_SECS) {
                return;
            }
        }
        self.gpu_last_sample = Some(Instant::now());
        match query_gpu_status() {
            Ok(status) => {
                self.gpu_status = Some(status);
                self.gpu_available = Some(true);
            }
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    self.gpu_available = Some(false);
                    self.gpu_status = None;
                } else {
                    if self.gpu_available.is_none() {
                        self.gpu_available = Some(true);
                    }
                    self.gpu_status = None;
                }
            }
        }
    }

    fn emit_blink_events(&mut self, shimmer_levels: &[f32], active_indices: &HashSet<usize>) {
        let Some(hub) = self.event_hub.as_ref() else {
            return;
        };
        if self.blink_last.len() != self.chunks.len() {
            self.blink_last = vec![false; self.chunks.len()];
        }
        let mode = match self.similarity_mode {
            SimilarityMode::None => "none",
            SimilarityMode::Browse => "browse",
            SimilarityMode::Groups => "groups",
            SimilarityMode::Query => "query",
        };
        let view = match self.source_view {
            SourceViewMode::Overview => "overview",
            SourceViewMode::Sorted => "sorted",
        };
        let term = self
            .active_term()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "all".to_string());
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|t| t.as_millis())
            .unwrap_or(0);
        let search = self.search_filter.clone().unwrap_or_default();
        let chunks = &self.chunks;
        let blink_last = &mut self.blink_last;
        for (idx, level) in shimmer_levels.iter().enumerate() {
            let flash = *level > 1.0;
            if flash && !blink_last[idx] && active_indices.contains(&idx) {
                let fragment = chunks
                    .get(idx)
                    .map(|chunk| snip(&chunk.text, BLINK_FRAGMENT_LIMIT))
                    .unwrap_or_default();
                let payload = serde_json::json!({
                    "event": "blink",
                    "chunk": idx + 1,
                    "level": level,
                    "mode": mode,
                    "view": view,
                    "term": term,
                    "search": search,
                    "fragment": fragment,
                    "timestamp_ms": timestamp,
                });
                hub.send(&payload.to_string());
            }
            blink_last[idx] = flash;
        }
    }

    fn tick_animation(&mut self) {
        if self.cursor_blink_last.elapsed() >= Duration::from_millis(500) {
            self.cursor_blink_on = !self.cursor_blink_on;
            self.cursor_blink_last = Instant::now();
        }
        if self.orbit_speed == 0.0 {
            return;
        }
        self.orbit_angle += self.orbit_speed;
        if self.orbit_angle > std::f32::consts::TAU {
            self.orbit_angle -= std::f32::consts::TAU;
        }
    }
}

fn clean_text(text: &str) -> String {
    text.to_string()
}

fn sanitize_for_embedding(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_space = false;
    for ch in text.chars() {
        if ch.is_ascii_graphic() {
            out.push(ch);
            last_was_space = false;
        } else if ch.is_ascii_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        }
    }
    let trimmed = out.trim().to_string();
    if trimmed.is_empty() {
        " ".to_string()
    } else {
        trimmed
    }
}

fn query_gpu_status() -> io::Result<GpuStatus> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "nvidia-smi failed",
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing gpu status",
        ));
    }
    let mut parts = line.split(',').map(|value| value.trim());
    let util = parts
        .next()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0);
    let mem_used = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let mem_total = parts
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    Ok(GpuStatus {
        util,
        mem_used,
        mem_total,
    })
}

fn embed_log_ready(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("model ready") || msg.contains("model loaded")
}

fn embed_log_loading(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("loading")
}

fn invert_log_ready(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("model loaded") || msg.contains("cached")
}

fn invert_log_loading(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("loading")
}

fn normalize_token(token: &str) -> String {
    let mut out = String::new();
    for ch in token.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn is_matchable_token(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }
    token.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn is_term_token(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }
    token.chars().all(|ch| ch.is_ascii_alphabetic())
}

fn build_corpus(text: &str) -> (Vec<String>, HashMap<char, Vec<usize>>, HashSet<String>) {
    let mut words = Vec::new();
    let mut index: HashMap<char, Vec<usize>> = HashMap::new();
    let mut set: HashSet<String> = HashSet::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start {
            let token = normalize_token(&text[s..idx]);
            if !token.is_empty() && set.insert(token.clone()) {
                let pos = words.len();
                if let Some(first) = token.chars().next() {
                    index.entry(first).or_default().push(pos);
                }
                words.push(token);
            }
            start = None;
        }
    }
    if let Some(s) = start {
        let token = normalize_token(&text[s..]);
        if !token.is_empty() && set.insert(token.clone()) {
            let pos = words.len();
            if let Some(first) = token.chars().next() {
                index.entry(first).or_default().push(pos);
            }
            words.push(token);
        }
    }

    (words, index, set)
}

fn build_term_ticker(
    invert_texts: &[Option<String>],
    invert_scores: &[Option<f32>],
    corpus_set: &HashSet<String>,
) -> Vec<TermCount> {
    let mut stats: HashMap<String, TermStats> = HashMap::new();
    for (idx, text) in invert_texts.iter().enumerate() {
        let score = invert_scores
            .get(idx)
            .and_then(|value| *value)
            .unwrap_or(0.0);
        if score < MIN_INVERSION_SCORE {
            continue;
        }
        let Some(text) = text.as_deref() else {
            continue;
        };
        let tokens: Vec<String> = collect_token_ranges(text)
            .into_iter()
            .map(|(_, _, token)| token)
            .collect();
        if tokens.is_empty() {
            continue;
        }
        let mut local: HashMap<String, usize> = HashMap::new();

        for token in &tokens {
            if token.is_empty() || is_stop_word(token) || !is_term_token(token) {
                continue;
            }
            if !corpus_set.contains(token) {
                continue;
            }
            *local.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, count) in local {
            let entry = stats.entry(term).or_insert(TermStats {
                count: 0,
                doc_freq: 0,
                max_chunk: 0,
            });
            entry.count += count;
            entry.doc_freq += 1;
            if count > entry.max_chunk {
                entry.max_chunk = count;
            }
        }
    }

    let mut terms: Vec<(String, TermStats)> = stats
        .into_iter()
        .filter(|(_, stat)| stat.doc_freq >= 2 || stat.count >= 3)
        .collect();
    terms.sort_by(|(term_a, stat_a), (term_b, stat_b)| {
        let weight_a = (stat_a.doc_freq as i64) * 100 + stat_a.count as i64;
        let weight_b = (stat_b.doc_freq as i64) * 100 + stat_b.count as i64;
        weight_b
            .cmp(&weight_a)
            .then_with(|| stat_b.count.cmp(&stat_a.count))
            .then_with(|| term_a.cmp(term_b))
    });

    let mut out: Vec<TermCount> = terms
        .into_iter()
        .map(|(term, stat)| TermCount {
            term,
            count: stat.doc_freq,
        })
        .collect();
    out.truncate(15);
    out.insert(
        0,
        TermCount {
            term: "All terms".to_string(),
            count: 0,
        },
    );
    out
}

fn string_width(text: &str) -> usize {
    text.chars().map(|ch| ch.width().unwrap_or(0)).sum()
}

fn score_term_in_text(term_tokens: &[String], text: &str) -> f32 {
    if term_tokens.is_empty() {
        return 0.0;
    }
    let tokens: Vec<String> = collect_token_ranges(text)
        .into_iter()
        .map(|(_, _, token)| token)
        .collect();
    if tokens.is_empty() {
        return 0.0;
    }
    if term_tokens.len() == 1 {
        let target = &term_tokens[0];
        let count = tokens.iter().filter(|token| *token == target).count();
        return count as f32;
    }
    let mut count = 0usize;
    let len = term_tokens.len();
    if tokens.len() < len {
        return 0.0;
    }
    for start in 0..=tokens.len() - len {
        if tokens[start..start + len] == term_tokens[..] {
            count += 1;
        }
    }
    count as f32
}

fn slice_by_width(text: &str, offset: usize, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let mut skipped = 0usize;
    let mut used = 0usize;
    let mut out = String::new();
    for ch in text.chars() {
        let width = ch.width().unwrap_or(0);
        if skipped + width <= offset {
            skipped += width;
            continue;
        }
        if used + width > max_width {
            break;
        }
        out.push(ch);
        used += width;
    }
    out
}

fn hash_embedding(embedding: &[f32]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for value in embedding {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{:016x}", hash)
}

fn hash_text(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

fn chunk_cache_key(text: &str) -> String {
    hash_text(&sanitize_for_embedding(text))
}

fn load_embedding_cache(window_tokens: usize) -> HashMap<String, Vec<f32>> {
    let path = PathBuf::from(EMBEDDING_CACHE_PATH);
    if !path.exists() {
        return HashMap::new();
    }
    let data = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(_) => return HashMap::new(),
    };
    let file: EmbeddingCacheFile = match serde_json::from_str(&data) {
        Ok(file) => file,
        Err(_) => return HashMap::new(),
    };
    if file.window_tokens != window_tokens {
        return HashMap::new();
    }
    file.entries
}

fn save_embedding_cache(window_tokens: usize, entries: &HashMap<String, Vec<f32>>) -> Result<()> {
    let path = PathBuf::from(EMBEDDING_CACHE_PATH);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache dir {}", parent.display()))?;
    }
    let file = EmbeddingCacheFile {
        window_tokens,
        entries: entries.clone(),
    };
    let data = serde_json::to_string(&file).context("serialize embedding cache")?;
    fs::write(&path, data).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_inversion_cache() -> HashMap<String, CachedInversion> {
    let path = PathBuf::from(INVERSION_CACHE_PATH);
    if !path.exists() {
        return HashMap::new();
    }
    let data = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(_) => return HashMap::new(),
    };
    serde_json::from_str(&data).unwrap_or_default()
}

fn save_inversion_cache(cache: &HashMap<String, CachedInversion>) -> Result<()> {
    let path = PathBuf::from(INVERSION_CACHE_PATH);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache dir {}", parent.display()))?;
    }
    let data = serde_json::to_string(cache).context("serialize inversion cache")?;
    fs::write(&path, data).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn collect_token_ranges(text: &str) -> Vec<(usize, usize, String)> {
    let mut ranges = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start {
            let token = normalize_token(&text[s..idx]);
            if !token.is_empty() {
                ranges.push((s, idx, token));
            }
            start = None;
        }
    }
    if let Some(s) = start {
        let token = normalize_token(&text[s..]);
        if !token.is_empty() {
            ranges.push((s, text.len(), token));
        }
    }
    ranges
}

fn levenshtein_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.is_empty() || b_chars.is_empty() {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0usize; b_chars.len() + 1];
    for (i, ac) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bc) in b_chars.iter().enumerate() {
            let cost = if ac == bc { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
        }
        prev.clone_from_slice(&curr);
    }
    let dist = prev[b_chars.len()];
    let max_len = a_chars.len().max(b_chars.len());
    1.0 - (dist as f32 / max_len as f32)
}

fn merge_fuzzy_matches(mut matches: Vec<FuzzyMatch>) -> Vec<FuzzyMatch> {
    if matches.is_empty() {
        return matches;
    }
    matches.sort_by(|a, b| (a.start, a.end).cmp(&(b.start, b.end)));
    let mut merged: Vec<FuzzyMatch> = Vec::new();
    for item in matches {
        if let Some(last) = merged.last_mut() {
            if last.start == item.start && last.end == item.end {
                if item.score > last.score {
                    last.score = item.score;
                }
                continue;
            }
        }
        merged.push(item);
    }
    merged
}

fn is_stop_word(token: &str) -> bool {
    STOP_WORDS.contains(&token)
}

fn compute_prune_boundary(text: &str, corpus_set: &HashSet<String>) -> Option<usize> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut start: Option<usize> = None;
    let mut boundary: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start {
            let token = normalize_token(&text[s..idx]);
            if !token.is_empty()
                && corpus_set.contains(&token)
                && !is_stop_word(&token)
                && !seen.contains(&token)
            {
                boundary = Some(idx);
                seen.insert(token);
            }
            start = None;
        }
    }

    if let Some(s) = start {
        let token = normalize_token(&text[s..]);
        if !token.is_empty()
            && corpus_set.contains(&token)
            && !is_stop_word(&token)
            && !seen.contains(&token)
        {
            boundary = Some(text.len());
        }
    }

    boundary
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
    let mut prev_end = 0usize;

    while idx < offsets.len() {
        let end_idx = (idx + tokens_per_chunk).min(offsets.len());
        let start = prev_end;
        let end = if end_idx < offsets.len() {
            offsets[end_idx].0
        } else {
            text.len()
        };
        let slice = text.get(start..end).unwrap_or("");
        seeds.push(ChunkSeed {
            start,
            end,
            text: slice.to_string(),
        });
        prev_end = end;
        idx = end_idx;
    }
    seeds
}

fn seed_chunks_from_tokens(text: &str, window_tokens: usize) -> SeedAnalysis {
    let seeds = build_token_seeds(text, window_tokens);
    let chunks = seeds
        .into_iter()
        .enumerate()
        .map(|(index, seed)| build_chunk(index, seed))
        .collect();
    SeedAnalysis { chunks }
}

fn build_chunk(index: usize, seed: ChunkSeed) -> ChunkInfo {
    let preview = snip(&seed.text, 80);
    ChunkInfo {
        index,
        start: seed.start,
        end: seed.end,
        preview,
        text: seed.text,
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

fn embedding_to_point(app: &mut App, embedding: &[f32]) -> [f32; 3] {
    ensure_projection_axes(app);
    if let Some(axes) = app.projection_axes.as_ref() {
        let x = dot_vec(&axes[0], embedding);
        let y = dot_vec(&axes[1], embedding);
        let z = dot_vec(&axes[2], embedding);
        return normalize_point([x, y, z]);
    }
    let (x, y, z) = if embedding.len() >= 3 {
        (embedding[0], embedding[1], embedding[2])
    } else {
        (0.0, 0.0, 0.0)
    };
    normalize_point([x, y, z])
}

fn normalize_point(point: [f32; 3]) -> [f32; 3] {
    let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    if norm <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [point[0] / norm, point[1] / norm, point[2] / norm]
    }
}

fn ensure_projection_axes(app: &mut App) {
    if app.projection_dirty || app.projection_axes.is_none() {
        app.projection_axes = build_projection_axes(app);
        app.projection_dirty = false;
    }
}

fn build_projection_axes(app: &App) -> Option<[Vec<f32>; 3]> {
    let embeddings: Vec<&Vec<f32>> = app.embeddings.iter().filter_map(|emb| emb.as_ref()).collect();
    if embeddings.len() < 2 {
        return None;
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return None;
    }

    let mut candidates: Vec<Vec<f32>> = Vec::new();
    let mut terms: Vec<String> = Vec::new();
    if !app.selected_terms.is_empty() {
        terms.extend(app.selected_terms.iter().cloned());
    } else if let Some(term) = app.active_term() {
        terms.push(term.to_string());
    }
    for term in terms.into_iter().take(3) {
        if let Some(axis) = build_term_axis(app, &term, dim) {
            candidates.push(axis);
        }
    }

    candidates.extend(build_global_axes(&embeddings, dim));
    let mut axes = orthonormalize_axes(&candidates);

    let mut seed = (dim as u64).wrapping_mul(0x9e3779b97f4a7c15);
    while axes.len() < 3 {
        let random = random_axis(dim, seed);
        axes.push(random);
        axes = orthonormalize_axes(&axes);
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        if seed == 0 {
            break;
        }
    }

    if axes.len() >= 3 {
        Some([axes[0].clone(), axes[1].clone(), axes[2].clone()])
    } else {
        None
    }
}

fn build_term_axis(app: &App, term: &str, dim: usize) -> Option<Vec<f32>> {
    let tokens: Vec<String> = term
        .split_whitespace()
        .map(normalize_token)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        return None;
    }
    let mut sum = vec![0.0f32; dim];
    let mut total = 0.0f32;
    for (idx, emb) in app.embeddings.iter().enumerate() {
        let Some(emb) = emb.as_ref() else {
            continue;
        };
        let score = term_score_for_chunk(app, &tokens, idx);
        if score <= 0.0 {
            continue;
        }
        for i in 0..dim.min(emb.len()) {
            sum[i] += emb[i] * score;
        }
        total += score;
    }
    if total <= f32::EPSILON {
        return None;
    }
    for value in &mut sum {
        *value /= total;
    }
    normalize_vec(sum)
}

fn term_score_for_chunk(app: &App, term_tokens: &[String], idx: usize) -> f32 {
    let score = app
        .invert_scores
        .get(idx)
        .and_then(|value| *value)
        .unwrap_or(0.0);
    if score < MIN_INVERSION_SCORE {
        return 0.0;
    }
    let Some(text) = app.invert_texts.get(idx).and_then(|value| value.as_deref()) else {
        return 0.0;
    };
    score_term_in_text(term_tokens, text)
}

fn build_global_axes(embeddings: &[&Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let mean = mean_embedding(embeddings, dim);
    let mut axes: Vec<Vec<f32>> = Vec::new();
    if let Some(axis0) = normalize_vec(mean.clone()) {
        axes.push(axis0);
    }

    let mut best_dist = 0.0f32;
    let mut axis1: Option<Vec<f32>> = None;
    for emb in embeddings {
        let diff = vector_sub(emb, &mean, dim);
        let dist = norm_vec(&diff);
        if dist > best_dist {
            best_dist = dist;
            axis1 = Some(diff);
        }
    }
    if let Some(axis) = axis1 {
        if let Some(normed) = normalize_vec(axis) {
            axes.push(normed);
        }
    }

    let mut best_residual = 0.0f32;
    let mut axis2: Option<Vec<f32>> = None;
    for emb in embeddings {
        let mut v = vector_sub(emb, &mean, dim);
        for axis in &axes {
            let proj = dot_vec(axis, &v);
            for i in 0..dim {
                v[i] -= proj * axis[i];
            }
        }
        let dist = norm_vec(&v);
        if dist > best_residual {
            best_residual = dist;
            axis2 = Some(v);
        }
    }
    if let Some(axis) = axis2 {
        if let Some(normed) = normalize_vec(axis) {
            axes.push(normed);
        }
    }

    axes
}

fn mean_embedding(embeddings: &[&Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for i in 0..dim.min(emb.len()) {
            mean[i] += emb[i];
        }
    }
    let denom = embeddings.len() as f32;
    if denom > 0.0 {
        for value in &mut mean {
            *value /= denom;
        }
    }
    mean
}

fn vector_sub(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for i in 0..dim.min(a.len()).min(b.len()) {
        out[i] = a[i] - b[i];
    }
    out
}

fn orthonormalize_axes(axes: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::new();
    for axis in axes {
        let mut v = axis.clone();
        for basis in &out {
            let proj = dot_vec(basis, &v);
            for i in 0..v.len().min(basis.len()) {
                v[i] -= proj * basis[i];
            }
        }
        if let Some(normed) = normalize_vec(v) {
            out.push(normed);
        }
        if out.len() >= 3 {
            break;
        }
    }
    out
}

fn normalize_vec(mut v: Vec<f32>) -> Option<Vec<f32>> {
    let norm = norm_vec(&v);
    if norm <= f32::EPSILON {
        return None;
    }
    for value in &mut v {
        *value /= norm;
    }
    Some(v)
}

fn norm_vec(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for value in v {
        sum += value * value;
    }
    sum.sqrt()
}

fn dot_vec(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn random_axis(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let value = ((state >> 33) as f32) / (u32::MAX as f32);
        out[i] = value * 2.0 - 1.0;
    }
    out
}

fn rotate_point(point: [f32; 3], angle: f32) -> [f32; 3] {
    let (x, y, z) = (point[0], point[1], point[2]);
    let tilt = 0.45f32;
    let cos_t = tilt.cos();
    let sin_t = tilt.sin();
    let x1 = x * cos_t - z * sin_t;
    let z1 = x * sin_t + z * cos_t;

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let x2 = x1 * cos_a + z1 * sin_a;
    let z2 = -x1 * sin_a + z1 * cos_a;
    [x2, y, z2]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn project_point_in_box(
    point: [f32; 3],
    front_x0: isize,
    front_y0: isize,
    front_x1: isize,
    front_y1: isize,
    back_x0: isize,
    back_y0: isize,
    back_x1: isize,
    back_y1: isize,
) -> Option<(usize, usize)> {
    let x = point[0].clamp(-1.0, 1.0);
    let y = point[1].clamp(-1.0, 1.0);
    let z = point[2].clamp(-1.0, 1.0);

    let x_t = (x + 1.0) * 0.5;
    let y_t = (y + 1.0) * 0.5;
    let z_t = (1.0 - z) * 0.5;

    let fx = lerp(front_x0 as f32, front_x1 as f32, x_t);
    let bx = lerp(back_x0 as f32, back_x1 as f32, x_t);
    let fy = lerp(front_y1 as f32, front_y0 as f32, y_t);
    let by = lerp(back_y1 as f32, back_y0 as f32, y_t);

    let sx = lerp(fx, bx, z_t).round() as isize;
    let sy = lerp(fy, by, z_t).round() as isize;

    if sx < 0 || sy < 0 {
        return None;
    }
    Some((sx as usize, sy as usize))
}

fn collect_vector_view_points(app: &mut App) -> Vec<VectorPoint> {
    if app.term_bias_active() {
        let mut scored: Vec<(usize, f32)> = app
            .term_scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut points = Vec::new();
        for (idx, _) in scored.into_iter().take(MAX_VECTOR_VIEW_POINTS) {
            if let Some(emb) = app
                .embeddings
                .get(idx)
                .and_then(|emb| emb.as_ref())
                .cloned()
            {
                points.push(VectorPoint {
                    index: idx,
                    coords: embedding_to_point(app, &emb),
                });
            }
        }
        if !points.is_empty() {
            return points;
        }
    }

    if app.similarity_mode == SimilarityMode::Query {
        let mut points = Vec::new();
        let query_indices: Vec<usize> = app
            .search_result_indices
            .iter()
            .copied()
            .take(MAX_QUERY_RESULTS)
            .collect();
        for idx in query_indices {
            if let Some(emb) = app
                .embeddings
                .get(idx)
                .and_then(|emb| emb.as_ref())
                .cloned()
            {
                points.push(VectorPoint {
                    index: idx,
                    coords: embedding_to_point(app, &emb),
                });
            }
        }
        return points;
    }

    if app.similarity_mode != SimilarityMode::Browse {
        let mut points = Vec::new();
        let auto_indices: Vec<usize> = app.auto_related_indices.clone();
        for idx in auto_indices {
            if let Some(emb) = app
                .embeddings
                .get(idx)
                .and_then(|emb| emb.as_ref())
                .cloned()
            {
                points.push(VectorPoint {
                    index: idx,
                    coords: embedding_to_point(app, &emb),
                });
            }
        }
        if !points.is_empty() {
            return points;
        }
        return app.vector_points.clone();
    }

    let mut scored: Vec<(usize, f32)> = app
        .similarity_scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| *idx != app.selected)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut points = Vec::new();
    for (idx, _score) in scored.into_iter().take(MAX_SIMILAR_VECTOR_POINTS) {
        if let Some(emb) = app
            .embeddings
            .get(idx)
            .and_then(|emb| emb.as_ref())
            .cloned()
        {
            points.push(VectorPoint {
                index: idx,
                coords: embedding_to_point(app, &emb),
            });
        }
    }
    points
}

fn centroid_point(points: &[VectorPoint]) -> [f32; 3] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut sum = [0.0f32; 3];
    for point in points {
        sum[0] += point.coords[0];
        sum[1] += point.coords[1];
        sum[2] += point.coords[2];
    }
    let denom = points.len() as f32;
    [sum[0] / denom, sum[1] / denom, sum[2] / denom]
}

fn build_scanner_points(app: &mut App) -> Vec<VectorPoint> {
    let mut view_points = collect_vector_view_points(app);
    let mut center_point = centroid_point(&view_points);
    if app.similarity_mode == SimilarityMode::Browse {
        if let Some(selected_emb) = app
            .embeddings
            .get(app.selected)
            .and_then(|emb| emb.as_ref())
            .cloned()
        {
            let selected_coords = embedding_to_point(app, &selected_emb);
            center_point = selected_coords;
            if !view_points.iter().any(|point| point.index == app.selected) {
                view_points.push(VectorPoint {
                    index: app.selected,
                    coords: selected_coords,
                });
            }
        }
    } else if app.similarity_mode == SimilarityMode::Query {
        if let Some(query_emb) = app.search_embedding.clone() {
            center_point = embedding_to_point(app, &query_emb);
        }
    }

    let mut centered_points: Vec<VectorPoint> = view_points
        .into_iter()
        .map(|point| VectorPoint {
            index: point.index,
            coords: [
                point.coords[0] - center_point[0],
                point.coords[1] - center_point[1],
                point.coords[2] - center_point[2],
            ],
        })
        .collect();

    let mut max_axis = [0.0f32; 3];
    for point in &centered_points {
        max_axis[0] = max_axis[0].max(point.coords[0].abs());
        max_axis[1] = max_axis[1].max(point.coords[1].abs());
        max_axis[2] = max_axis[2].max(point.coords[2].abs());
    }
    let max_xz = max_axis[0].max(max_axis[2]);
    let max_scale = 2.0;
    let base_scale_xz = if max_xz > f32::EPSILON {
        (0.7 / max_xz).min(max_scale)
    } else {
        1.0
    };
    let base_scale_y = if max_axis[1] > f32::EPSILON {
        (0.9 / max_axis[1]).min(max_scale)
    } else {
        base_scale_xz
    };
    let scale_cap = max_scale * SCANNER_ZOOM_MAX;
    let scale_xz = (base_scale_xz * app.scanner_zoom).clamp(0.2, scale_cap);
    let scale_y = (base_scale_y * app.scanner_zoom).clamp(0.2, scale_cap);

    for point in &mut centered_points {
        point.coords[0] *= scale_xz;
        point.coords[1] *= scale_y;
        point.coords[2] *= scale_xz;
    }

    centered_points
}

fn build_shimmer_levels(app: &App, points: &[VectorPoint]) -> Vec<f32> {
    let mut levels = vec![0.0f32; app.chunks.len()];
    if !app.time_enabled {
        return levels;
    }
    let threshold = 0.35f32;
    let flash_threshold = 0.05f32;

    for point in points {
        if point.index >= levels.len() {
            continue;
        }
        let rotated = rotate_point(point.coords, app.orbit_angle);
        let dist = rotated[0].abs();
        let level = if dist <= flash_threshold {
            1.2
        } else if dist <= threshold {
            let closeness = 1.0 - (dist / threshold);
            (closeness * closeness).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if level > levels[point.index] {
            levels[point.index] = level;
        }
    }

    levels
}

fn set_cell(grid: &mut [Vec<VectorCell>], x: usize, y: usize, ch: char, style: Option<Style>) {
    if let Some(row) = grid.get_mut(y) {
        if let Some(cell) = row.get_mut(x) {
            *cell = VectorCell { ch, style };
        }
    }
}

fn set_cell_if_empty(
    grid: &mut [Vec<VectorCell>],
    x: usize,
    y: usize,
    ch: char,
    style: Option<Style>,
) {
    if let Some(row) = grid.get_mut(y) {
        if let Some(cell) = row.get_mut(x) {
            if cell.ch == ' ' {
                *cell = VectorCell { ch, style };
            }
        }
    }
}

fn set_label(grid: &mut [Vec<VectorCell>], x: usize, y: usize, label: &str, style: Option<Style>) {
    let mut xi = x;
    for ch in label.chars() {
        if let Some(row) = grid.get_mut(y) {
            if xi < row.len() {
                row[xi] = VectorCell { ch, style };
            }
        }
        xi += 1;
    }
}

fn draw_line(
    grid: &mut [Vec<VectorCell>],
    x0: isize,
    y0: isize,
    x1: isize,
    y1: isize,
    ch: char,
    style: Option<Style>,
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 {
            set_cell(grid, x0 as usize, y0 as usize, ch, style);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn connector_char(x0: isize, y0: isize, x1: isize, y1: isize) -> char {
    let dx = x1 - x0;
    let dy = y1 - y0;
    if dx == 0 {
        '|'
    } else if dy == 0 {
        '-'
    } else if (dx > 0) == (dy > 0) {
        '\\'
    } else {
        '/'
    }
}

fn draw_rect(
    grid: &mut [Vec<VectorCell>],
    x0: isize,
    y0: isize,
    x1: isize,
    y1: isize,
    ch: char,
    corner: char,
    style: Option<Style>,
) {
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    draw_line(grid, x0, y0, x1, y0, ch, style);
    draw_line(grid, x1, y0, x1, y1, ch, style);
    draw_line(grid, x1, y1, x0, y1, ch, style);
    draw_line(grid, x0, y1, x0, y0, ch, style);

    if x0 >= 0 && y0 >= 0 {
        set_cell(grid, x0 as usize, y0 as usize, corner, style);
    }
    if x1 >= 0 && y0 >= 0 {
        set_cell(grid, x1 as usize, y0 as usize, corner, style);
    }
    if x1 >= 0 && y1 >= 0 {
        set_cell(grid, x1 as usize, y1 as usize, corner, style);
    }
    if x0 >= 0 && y1 >= 0 {
        set_cell(grid, x0 as usize, y1 as usize, corner, style);
    }
}

fn grid_to_lines(grid: Vec<Vec<VectorCell>>) -> Vec<Line<'static>> {
    let mut lines = Vec::with_capacity(grid.len());
    for row in grid {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut current_style: Option<Style> = None;
        let mut buf = String::new();

        for cell in row {
            if cell.style != current_style {
                if !buf.is_empty() {
                    let span: Span<'static> = match current_style {
                        Some(style) => Span::styled(buf.clone(), style),
                        None => Span::raw(buf.clone()),
                    };
                    spans.push(span);
                    buf.clear();
                }
                current_style = cell.style;
            }
            buf.push(cell.ch);
        }

        if !buf.is_empty() {
            let span: Span<'static> = match current_style {
                Some(style) => Span::styled(buf, style),
                None => Span::raw(buf),
            };
            spans.push(span);
        }

        if spans.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(spans));
        }
    }
    lines
}

fn apply_depth_tint(rgb: (u8, u8, u8), depth: f32) -> Color {
    let (r, g, b) = rgb;
    let t = ((depth + 1.0) * 0.5).clamp(0.0, 1.0);
    let factor = 0.6 + t * 0.4;
    let scale = |v: u8| ((v as f32 * factor).clamp(0.0, 255.0)) as u8;
    Color::Rgb(scale(r), scale(g), scale(b))
}

fn build_vector_view_lines(app: &mut App, width: u16, height: u16) -> Vec<Line<'static>> {
    let inner_w = width.saturating_sub(2) as usize;
    let inner_h = height.saturating_sub(2) as usize;
    if inner_w == 0 || inner_h == 0 {
        return vec![Line::from("")];
    }

    let mut grid = vec![
        vec![
            VectorCell {
                ch: ' ',
                style: None,
            };
            inner_w
        ];
        inner_h
    ];

    let mut inset = inner_w.min(inner_h).saturating_sub(2).min(3) as isize;
    if inset < 1 {
        inset = 1;
    }
    let front_x0 = 0;
    let front_y0 = 0;
    let front_x1 = inner_w.saturating_sub(1) as isize;
    let front_y1 = inner_h.saturating_sub(1) as isize;
    let back_x0 = (front_x0 + inset).min(front_x1);
    let back_y0 = (front_y0 + inset).min(front_y1);
    let back_x1 = (front_x1 - inset).max(back_x0);
    let back_y1 = (front_y1 - inset).max(back_y0);

    let back_style = Style::default().fg(Color::Gray);
    let edge_style = Style::default().fg(Color::DarkGray);
    let corner_char = if app.time_enabled {
        let spinner = ['|', '/', '-', '\\'];
        let spin_phase = app.orbit_angle * 6.0;
        let step = std::f32::consts::TAU / spinner.len() as f32;
        let idx = ((spin_phase / step) as usize) % spinner.len();
        spinner[idx]
    } else {
        '\\'
    };

    draw_line(
        &mut grid,
        back_x0,
        back_y0,
        back_x1,
        back_y0,
        '-',
        Some(back_style),
    );
    draw_line(
        &mut grid,
        back_x0,
        back_y1,
        back_x1,
        back_y1,
        '-',
        Some(back_style),
    );
    draw_line(
        &mut grid,
        back_x0,
        back_y0,
        back_x0,
        back_y1,
        '|',
        Some(back_style),
    );
    draw_line(
        &mut grid,
        back_x1,
        back_y0,
        back_x1,
        back_y1,
        '|',
        Some(back_style),
    );
    set_cell(&mut grid, back_x0 as usize, back_y0 as usize, corner_char, Some(back_style));
    set_cell(&mut grid, back_x1 as usize, back_y0 as usize, corner_char, Some(back_style));
    set_cell(&mut grid, back_x1 as usize, back_y1 as usize, corner_char, Some(back_style));
    set_cell(&mut grid, back_x0 as usize, back_y1 as usize, corner_char, Some(back_style));

    draw_line(
        &mut grid,
        front_x0,
        front_y0,
        back_x0,
        back_y0,
        connector_char(front_x0, front_y0, back_x0, back_y0),
        Some(edge_style),
    );
    draw_line(
        &mut grid,
        front_x1,
        front_y0,
        back_x1,
        back_y0,
        connector_char(front_x1, front_y0, back_x1, back_y0),
        Some(edge_style),
    );
    draw_line(
        &mut grid,
        front_x1,
        front_y1,
        back_x1,
        back_y1,
        connector_char(front_x1, front_y1, back_x1, back_y1),
        Some(edge_style),
    );
    draw_line(
        &mut grid,
        front_x0,
        front_y1,
        back_x0,
        back_y1,
        connector_char(front_x0, front_y1, back_x0, back_y1),
        Some(edge_style),
    );

    let centered_points = build_scanner_points(app);

    let center_marker = rotate_point([0.0, 0.0, 0.0], app.orbit_angle);
    if let Some((gx, gy)) = project_point_in_box(
        center_marker,
        front_x0,
        front_y0,
        front_x1,
        front_y1,
        back_x0,
        back_y0,
        back_x1,
        back_y1,
    ) {
        if gx < inner_w && gy < inner_h {
            let style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
            set_cell_if_empty(&mut grid, gx, gy, 'o', Some(style));
        }
    }

    let floor_y = -1.0;
    for point in centered_points {
        let rotated = rotate_point(point.coords, app.orbit_angle);
        let shadow_rotated = rotate_point([point.coords[0], floor_y, point.coords[2]], app.orbit_angle);

        if let Some((sx, sy)) = project_point_in_box(
            shadow_rotated,
            front_x0,
            front_y0,
            front_x1,
            front_y1,
            back_x0,
            back_y0,
            back_x1,
            back_y1,
        ) {
            if sx < inner_w && sy < inner_h {
                let base_rgb = chunk_palette(point.index);
                let shadow_color = apply_depth_tint(base_rgb, shadow_rotated[2]);
                let style = Style::default().fg(shadow_color);
                set_cell_if_empty(&mut grid, sx, sy, '.', Some(style));
            }
        }

        if let Some((gx, gy)) = project_point_in_box(
            rotated,
            front_x0,
            front_y0,
            front_x1,
            front_y1,
            back_x0,
            back_y0,
            back_x1,
            back_y1,
        ) {
            if gx < inner_w && gy < inner_h {
                let label = format!("{}", point.index + 1);
                let base_rgb = chunk_palette(point.index);
                let color = apply_depth_tint(base_rgb, rotated[2]);
                let style = Style::default().fg(color).add_modifier(Modifier::BOLD);
                set_label(&mut grid, gx, gy, &label, Some(style));
            }
        }
    }

    grid_to_lines(grid)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}

fn chunk_palette(index: usize) -> (u8, u8, u8) {
    // Outline: HSL palette seeded by golden angle for 100+ distinct colors.
    let hue = (index as f32 * 137.5) % 360.0;
    hsl_to_rgb(hue, 0.65, 0.52)
}

fn group_palette(group_id: usize) -> (u8, u8, u8) {
    let hue = (group_id as f32 * 137.5 + 25.0) % 360.0;
    hsl_to_rgb(hue, 0.72, 0.48)
}

fn apply_similarity_tint(rgb: (u8, u8, u8), similarity: f32, dim: bool) -> Color {
    let (r, g, b) = rgb;
    let sim = similarity.clamp(0.0, 1.0);
    let factor = if dim {
        if sim > 0.85 {
            1.2
        } else if sim > 0.65 {
            1.0
        } else if sim > 0.45 {
            0.75
        } else {
            0.55
        }
    } else {
        1.0
    };
    let scale = |v: u8| ((v as f32 * factor).clamp(0.0, 255.0)) as u8;
    Color::Rgb(scale(r), scale(g), scale(b))
}

fn chunk_base_color(app: &App, chunk: &ChunkInfo) -> Color {
    let base_rgb = chunk_palette(chunk.index);
    match app.similarity_mode {
        SimilarityMode::Browse | SimilarityMode::Query => {
            let similarity = app
                .similarity_scores
                .get(chunk.index)
                .copied()
                .unwrap_or(0.0);
            apply_similarity_tint(base_rgb, similarity, true)
        }
        SimilarityMode::Groups => {
            if let Some(group_id) = app.group_ids.get(chunk.index).and_then(|id| *id) {
                let (r, g, b) = group_palette(group_id);
                Color::Rgb(r, g, b)
            } else {
                Color::Rgb(base_rgb.0, base_rgb.1, base_rgb.2)
            }
        }
        SimilarityMode::None => Color::Rgb(base_rgb.0, base_rgb.1, base_rgb.2),
    }
}

fn apply_shimmer_color(color: Color, intensity: f32, flash: bool) -> Color {
    if let Color::Rgb(r, g, b) = color {
        let boost = if flash { 1.6 } else { 1.0 + 0.8 * intensity };
        let scale = |v: u8| ((v as f32 * boost).clamp(0.0, 255.0)) as u8;
        Color::Rgb(scale(r), scale(g), scale(b))
    } else {
        color
    }
}

fn chunk_style(app: &App, chunk: &ChunkInfo, selected: bool) -> Style {
    let color = chunk_base_color(app, chunk);
    let mut style = Style::default().fg(color);
    if app
        .invert_done
        .get(chunk.index)
        .copied()
        .unwrap_or(false)
    {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if selected {
        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
    }
    style
}

fn chunk_style_with_shimmer(app: &App, chunk: &ChunkInfo, selected: bool, shimmer: f32) -> Style {
    let base_color = chunk_base_color(app, chunk);
    let flash = shimmer > 1.0;
    let intensity = shimmer.min(1.0);
    let color = if shimmer > 0.0 {
        apply_shimmer_color(base_color, intensity, flash)
    } else {
        base_color
    };
    let mut style = Style::default().fg(color);
    if shimmer > 0.0 {
        style = style.add_modifier(Modifier::BOLD);
        if flash {
            style = style.add_modifier(Modifier::REVERSED);
        }
    }
    if app
        .invert_done
        .get(chunk.index)
        .copied()
        .unwrap_or(false)
    {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if selected {
        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
    }
    style
}

fn list_chunk_style(app: &App, chunk: &ChunkInfo, selected: bool) -> Style {
    chunk_style(app, chunk, selected)
}

fn flush_span<'a>(spans: &mut Vec<Span<'a>>, text: &mut String, style: &mut Option<Style>) {
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
    show_markers: bool,
) -> (Vec<Line<'a>>, Option<usize>) {
    if max_width == 0 {
        return (lines, None);
    }

    let mut wrapped: Vec<Line> = Vec::new();
    let mut marker_count = 0usize;
    let mut selected_line = None;

    for line in lines {
        let mut current_spans: Vec<Span> = Vec::new();
        let mut current_span_text = String::new();
        let mut current_span_style: Option<Style> = None;
        let mut current_width = 0usize;
        let mut line_had_content = false;
        let mut pushed_any = false;

        for span in line.spans {
            let span_style = span.style;
            for ch in span.content.chars() {
                if ch == '«' {
                    if let Some(target) = target_marker {
                        if marker_count == target && selected_line.is_none() {
                            selected_line = Some(wrapped.len());
                        }
                    }
                    marker_count += 1;
                    if !show_markers {
                        continue;
                    }
                }
                if ch == '»' && !show_markers {
                    continue;
                }
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

fn push_span<'a>(spans: &mut Vec<Span<'a>>, text: String, style: Style) {
    spans.push(Span::styled(text, style));
}

fn append_text<'a>(lines: &mut Vec<Line<'a>>, spans: &mut Vec<Span<'a>>, text: &str, style: Style) {
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

fn append_text_with_highlight<'a>(
    lines: &mut Vec<Line<'a>>,
    spans: &mut Vec<Span<'a>>,
    text: &str,
    text_start: usize,
    base_style: Style,
    show_markers: bool,
    search_matches: &[(usize, usize)],
    fuzzy_matches: &[FuzzyMatch],
) {
    #[derive(PartialEq)]
    enum HighlightKind {
        None,
        Search,
        Fuzzy { intensity: u8, strong: bool },
    }

    let highlight_style = Style::default().fg(Color::Black).bg(Color::Yellow);
    let text_end = text_start + text.len();

    let relevant_search: Vec<(usize, usize)> = search_matches
        .iter()
        .filter(|(ms, me)| *ms < text_end && *me > text_start)
        .cloned()
        .collect();
    let relevant_fuzzy: Vec<FuzzyMatch> = fuzzy_matches
        .iter()
        .filter(|m| m.start < text_end && m.end > text_start)
        .cloned()
        .collect();

    if relevant_search.is_empty() && relevant_fuzzy.is_empty() {
        append_text(lines, spans, text, base_style);
        return;
    }

    let chars: Vec<char> = text.chars().collect();
    let mut char_offsets: Vec<usize> = Vec::with_capacity(chars.len() + 1);
    let mut byte_offset = 0;
    for ch in &chars {
        char_offsets.push(byte_offset);
        byte_offset += ch.len_utf8();
    }
    char_offsets.push(byte_offset);

    let mut buf = String::new();
    let mut current_kind = HighlightKind::None;
    let mut last_was_cr = false;

    let is_fuzzy = |kind: &HighlightKind| matches!(kind, HighlightKind::Fuzzy { .. });
    let is_strong = |kind: &HighlightKind| matches!(kind, HighlightKind::Fuzzy { strong: true, .. });
    let kind_style = |kind: &HighlightKind| match kind {
        HighlightKind::None => base_style,
        HighlightKind::Search => highlight_style,
        HighlightKind::Fuzzy { intensity, strong } => {
            let mut style = base_style
                .fg(Color::Black)
                .bg(Color::Rgb(255, *intensity, 80))
                .add_modifier(Modifier::BOLD);
            if *strong {
                style = style.add_modifier(Modifier::UNDERLINED);
            }
            style
        }
    };
    let push_fuzzy_marker = |spans: &mut Vec<Span<'a>>, kind: &HighlightKind, open: bool| {
        if !show_markers || !is_fuzzy(kind) {
            return;
        }
        if open {
            push_span(spans, "`".to_string(), kind_style(kind));
            if is_strong(kind) {
                push_span(spans, "**".to_string(), kind_style(kind));
            }
        } else {
            if is_strong(kind) {
                push_span(spans, "**".to_string(), kind_style(kind));
            }
            push_span(spans, "`".to_string(), kind_style(kind));
        }
    };

    for (i, &ch) in chars.iter().enumerate() {
        let global_pos = text_start + char_offsets[i];
        let in_search = relevant_search
            .iter()
            .any(|(ms, me)| global_pos >= *ms && global_pos < *me);
        let mut fuzzy_score = 0.0f32;
        let mut fuzzy_strong = false;
        if !in_search {
            for m in &relevant_fuzzy {
                if global_pos >= m.start && global_pos < m.end {
                    if m.score > fuzzy_score {
                        fuzzy_score = m.score;
                    }
                    if m.score > 1.0 {
                        fuzzy_strong = true;
                    }
                }
            }
        }
        let new_kind = if in_search {
            HighlightKind::Search
        } else if fuzzy_score > 0.0 {
            let intensity = (fuzzy_score.clamp(0.0, 1.0) * 75.0 + 180.0) as u8;
            HighlightKind::Fuzzy {
                intensity,
                strong: fuzzy_strong,
            }
        } else {
            HighlightKind::None
        };

        if ch == '\r' || ch == '\n' {
            if !buf.is_empty() {
                push_span(spans, buf.clone(), kind_style(&current_kind));
                buf.clear();
            }
            if is_fuzzy(&current_kind) {
                push_fuzzy_marker(spans, &current_kind, false);
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
            current_kind = HighlightKind::None;
            continue;
        }
        last_was_cr = false;

        if new_kind != current_kind {
            if !buf.is_empty() {
                push_span(spans, buf.clone(), kind_style(&current_kind));
                buf.clear();
            }
            if is_fuzzy(&current_kind) {
                push_fuzzy_marker(spans, &current_kind, false);
            }
            if is_fuzzy(&new_kind) {
                push_fuzzy_marker(spans, &new_kind, true);
            }
            current_kind = new_kind;
        }
        buf.push(ch);
    }

    if !buf.is_empty() {
        push_span(spans, buf, kind_style(&current_kind));
    }
    if is_fuzzy(&current_kind) {
        push_fuzzy_marker(spans, &current_kind, false);
    }
}

fn build_overview_lines(app: &mut App) -> Vec<Line<'static>> {
    if app.text.is_empty() {
        return vec![Line::from("no text loaded")];
    }

    let scanner_points = build_scanner_points(app);
    let shimmer_levels = build_shimmer_levels(app, &scanner_points);

    let mut lines: Vec<Line> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut cursor = 0usize;

    let mut chunks = app.chunks.clone();
    chunks.sort_by_key(|chunk| chunk.start);

    let default_marker_style = Style::default().fg(Color::White).bg(Color::DarkGray);
    let done_marker_style = Style::default().fg(Color::Black).bg(Color::Green);
    let search_matches = &app.search_matches;
    let empty_fuzzy: &[FuzzyMatch] = &[];

    for chunk in chunks {
        let start = chunk.start.min(app.text.len());
        let end = chunk.end.min(app.text.len());
        if start > cursor {
            if let Some(gap) = app.text.get(cursor..start) {
                append_text_with_highlight(
                    &mut lines,
                    &mut spans,
                    gap,
                    cursor,
                    Style::default().fg(Color::Gray),
                    app.show_markers,
                    search_matches,
                    empty_fuzzy,
                );
            }
        }
        if end > start {
            let shimmer = shimmer_levels.get(chunk.index).copied().unwrap_or(0.0);
            let style = chunk_style_with_shimmer(app, &chunk, app.selected == chunk.index, shimmer);
            let marker_style = if app
                .invert_done
                .get(chunk.index)
                .copied()
                .unwrap_or(false)
            {
                done_marker_style
            } else {
                default_marker_style
            };
            let label = format!("{:03} ", chunk.index + 1);
            let chunk_fuzzy = app
                .invert_matches
                .get(chunk.index)
                .map(|matches| matches.as_slice())
                .unwrap_or(empty_fuzzy);
            push_span(&mut spans, "«".to_string(), marker_style);
            if app.show_markers {
                push_span(&mut spans, label, marker_style);
            }
            if let Some(segment) = app.text.get(start..end) {
                append_text_with_highlight(
                    &mut lines,
                    &mut spans,
                    segment,
                    start,
                    style,
                    app.show_markers,
                    search_matches,
                    chunk_fuzzy,
                );
            }
            push_span(&mut spans, "»".to_string(), marker_style);
        }
        cursor = end;
    }

    if cursor < app.text.len() {
        if let Some(rest) = app.text.get(cursor..) {
            append_text_with_highlight(
                &mut lines,
                &mut spans,
                rest,
                cursor,
                Style::default().fg(Color::Gray),
                app.show_markers,
                search_matches,
                empty_fuzzy,
            );
        }
    }

    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }

    lines
}

fn build_sorted_lines(app: &mut App, list_indices: &[usize]) -> Vec<Line<'static>> {
    if app.text.is_empty() {
        return vec![Line::from("no text loaded")];
    }
    if list_indices.is_empty() {
        return vec![Line::from("no chunks to display")];
    }

    let scanner_points = build_scanner_points(app);
    let shimmer_levels = build_shimmer_levels(app, &scanner_points);
    let default_marker_style = Style::default().fg(Color::White).bg(Color::DarkGray);
    let done_marker_style = Style::default().fg(Color::Black).bg(Color::Green);
    let search_matches = &app.search_matches;
    let empty_fuzzy: &[FuzzyMatch] = &[];

    let mut lines: Vec<Line> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();

    for idx in list_indices {
        let Some(chunk) = app.chunks.get(*idx) else {
            continue;
        };
        let start = chunk.start.min(app.text.len());
        let end = chunk.end.min(app.text.len());
        if end <= start {
            continue;
        }

        let shimmer = shimmer_levels.get(chunk.index).copied().unwrap_or(0.0);
        let style = chunk_style_with_shimmer(app, chunk, app.selected == chunk.index, shimmer);
        let marker_style = if app
            .invert_done
            .get(chunk.index)
            .copied()
            .unwrap_or(false)
        {
            done_marker_style
        } else {
            default_marker_style
        };
        let label = format!("{:03} ", chunk.index + 1);
        let chunk_fuzzy = app
            .invert_matches
            .get(chunk.index)
            .map(|matches| matches.as_slice())
            .unwrap_or(empty_fuzzy);

        push_span(&mut spans, "«".to_string(), marker_style);
        if app.show_markers {
            push_span(&mut spans, label, marker_style);
        }
        if let Some(segment) = app.text.get(start..end) {
            append_text_with_highlight(
                &mut lines,
                &mut spans,
                segment,
                start,
                style,
                app.show_markers,
                search_matches,
                chunk_fuzzy,
            );
        }
        push_span(&mut spans, "»".to_string(), marker_style);
    }

    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    if lines.is_empty() {
        lines.push(Line::from("no chunks to display"));
    }

    lines
}

fn build_sorted_text(app: &App, list_indices: &[usize]) -> String {
    let mut out = String::new();
    for (pos, idx) in list_indices.iter().enumerate() {
        let Some(chunk) = app.chunks.get(*idx) else {
            continue;
        };
        if pos > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&chunk.text);
    }
    out
}

fn copy_with_command(cmd: &str, args: &[&str], text: &str) -> Result<()> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {}", cmd))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .with_context(|| format!("write {}", cmd))?;
    }
    let status = child.wait().with_context(|| format!("wait {}", cmd))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("{} failed", cmd))
    }
}

fn copy_to_clipboard(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        return copy_with_command("cmd", &["/C", "clip"], text);
    }
    #[cfg(target_os = "macos")]
    {
        return copy_with_command("pbcopy", &[], text);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let candidates: [(&str, &[&str]); 3] = [
            ("wl-copy", &[]),
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
        ];
        for (cmd, args) in candidates {
            if copy_with_command(cmd, args, text).is_ok() {
                return Ok(());
            }
        }
        return Err(anyhow!("no clipboard command succeeded"));
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
    {
        return Err(anyhow!("clipboard not supported on this platform"));
    }
}

fn build_extract_lines(text: &str) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![Line::from("")];
    }
    text
        .lines()
        .map(|line| Line::from(line.to_string()))
        .collect()
}

fn build_search_display(app: &App) -> String {
    let search_active = app.focus_area == FocusArea::Search;
    if !search_active && app.search_query.is_empty() {
        return "search \\ or /".to_string();
    }
    let mut search_display = app.search_query.clone();
    let display_len = search_display.chars().count();
    if display_len > SEARCH_BOX_WIDTH {
        search_display = search_display.chars().take(SEARCH_BOX_WIDTH).collect();
    } else if search_active && display_len < SEARCH_BOX_WIDTH {
        search_display.push_str(&" ".repeat(SEARCH_BOX_WIDTH - display_len));
    }
    search_display
}

fn build_search_spans(app: &App, search_style: Style) -> Vec<Span<'static>> {
    let search_active = app.focus_area == FocusArea::Search;
    if !search_active && app.search_query.is_empty() {
        return vec![Span::styled(
            "search \\ or /",
            Style::default().fg(Color::DarkGray),
        )];
    }
    if !search_active {
        let mut display = app.search_query.clone();
        if display.chars().count() > SEARCH_BOX_WIDTH {
            display = display.chars().take(SEARCH_BOX_WIDTH).collect();
        }
        return vec![Span::styled(display, search_style)];
    }

    let mut chars: Vec<char> = app.search_query.chars().collect();
    if chars.len() > SEARCH_BOX_WIDTH {
        chars.truncate(SEARCH_BOX_WIDTH);
    }
    if chars.len() < SEARCH_BOX_WIDTH {
        chars.extend(std::iter::repeat(' ').take(SEARCH_BOX_WIDTH - chars.len()));
    }

    let cursor_pos = app
        .search_query
        .chars()
        .count()
        .min(SEARCH_BOX_WIDTH.saturating_sub(1));
    let cursor_char = chars.get(cursor_pos).copied().unwrap_or(' ');
    let prefix: String = chars.iter().take(cursor_pos).collect();
    let suffix: String = chars.iter().skip(cursor_pos + 1).collect();

    let cursor_style = if app.cursor_blink_on {
        Style::default()
            .fg(Color::Black)
            .bg(Color::White)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        search_style
    };

    let mut spans = Vec::new();
    if !prefix.is_empty() {
        spans.push(Span::styled(prefix, search_style));
    }
    spans.push(Span::styled(cursor_char.to_string(), cursor_style));
    if !suffix.is_empty() {
        spans.push(Span::styled(suffix, search_style));
    }
    spans
}

fn build_gpu_spans(app: &App) -> Vec<Span<'static>> {
    let label_style = Style::default().fg(Color::DarkGray);
    let on_style = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
    let off_style = Style::default().fg(Color::DarkGray);
    let bracket_style = Style::default().fg(Color::DarkGray);
    let value_style = Style::default().fg(Color::Green);

    let mut spans = Vec::new();
    spans.push(Span::styled("GPU ", label_style));
    spans.push(Span::styled("[", bracket_style));

    let segments = 10usize;
    let util = app.gpu_status.map(|status| status.util).unwrap_or(0);
    let filled = ((util as usize * segments) + 99) / 100;

    for idx in 0..segments {
        let (ch, style) = if idx < filled {
            ('|', on_style)
        } else {
            ('.', off_style)
        };
        spans.push(Span::styled(ch.to_string(), style));
    }

    spans.push(Span::styled("]", bracket_style));

    if let Some(status) = app.gpu_status {
        spans.push(Span::styled(format!(" {:>3}%", status.util), value_style));
    } else if app.gpu_available == Some(false) {
        spans.push(Span::styled(" n/a", label_style));
    } else {
        spans.push(Span::styled(" --", label_style));
    }

    spans
}

fn line_to_string(line: &Line) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn build_list_snapshot_lines(app: &App, list_indices: &[usize], height: usize) -> Vec<String> {
    if list_indices.is_empty() {
        return vec!["no chunks".to_string()];
    }
    let list_len = list_indices.len();
    let visible = height.max(1).min(list_len);
    let selected_pos = list_indices
        .iter()
        .position(|idx| *idx == app.selected)
        .unwrap_or(0);
    let mut start = selected_pos.saturating_sub(visible / 2);
    if start + visible > list_len {
        start = list_len.saturating_sub(visible);
    }
    let end = (start + visible).min(list_len);

    let mut lines = Vec::new();
    for (pos, idx) in list_indices.iter().enumerate().take(end).skip(start) {
        let Some(chunk) = app.chunks.get(*idx) else {
            continue;
        };
        let sim = app
            .similarity_scores
            .get(chunk.index)
            .copied()
            .unwrap_or(0.0);
        let sim_suffix = match app.similarity_mode {
            SimilarityMode::Browse | SimilarityMode::Query => format!(" sim:{:.2}", sim),
            _ => String::new(),
        };
        let prune_suffix = if app.prune_available(chunk.index) {
            if app
                .invert_prune_marked
                .get(chunk.index)
                .copied()
                .unwrap_or(false)
            {
                " !"
            } else {
                " *"
            }
        } else {
            ""
        };
        let line = format!(
            "{:>3} [{:>5}-{:>5}] {}{}{}",
            chunk.index + 1,
            chunk.start,
            chunk.end,
            chunk.preview,
            sim_suffix,
            prune_suffix
        );
        if pos == selected_pos {
            lines.push(format!("▶ {}", line));
        } else {
            lines.push(format!("  {}", line));
        }
    }
    lines
}

fn build_screen_snapshot(app: &mut App) -> Result<String> {
    let (width, height) = crossterm::terminal::size().context("terminal size")?;
    let size = Rect {
        x: 0,
        y: 0,
        width,
        height,
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(size);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[2]);

    let use_query_list = app.similarity_mode == SimilarityMode::Query
        && !app.search_result_indices.is_empty();
    let use_browse_list = app.similarity_mode == SimilarityMode::Browse
        && !app.browse_result_indices.is_empty();
    let list_indices = current_list_indices(app);
    let source_indices: Vec<usize> = if app.source_view == SourceViewMode::Sorted
        && (use_query_list || use_browse_list)
    {
        list_indices
            .iter()
            .cloned()
            .take(FILTERED_TEXT_LIMIT)
            .collect()
    } else {
        list_indices.clone()
    };
    let list_title = if app.similarity_mode == SimilarityMode::Query {
        if let Some(filter) = &app.search_filter {
            format!("Chunks | filtering with \"{}\"", filter)
        } else {
            "Chunks".to_string()
        }
    } else if app.similarity_mode == SimilarityMode::Browse {
        let anchor = app.browse_anchor.unwrap_or(app.selected);
        format!("Chunks | similar to {:03}", anchor + 1)
    } else {
        "Chunks".to_string()
    };

    let inner_width = body[0].width.saturating_sub(2) as usize;
    let visible_height = body[0].height.saturating_sub(2).max(1) as usize;
    let (source_lines, source_title) = match app.source_view {
        SourceViewMode::Overview => (build_overview_lines(app), "Source Text"),
        SourceViewMode::Sorted => {
            let title = if use_query_list || use_browse_list {
                "Filtered Text"
            } else {
                "Sorted Text"
            };
            (build_sorted_lines(app, &source_indices), title)
        }
    };

    let (wrapped_lines, _) = wrap_lines_with_marker(
        source_lines,
        inner_width.max(1),
        None,
        app.show_markers,
    );
    let max_scroll = wrapped_lines.len().saturating_sub(visible_height);
    let scroll = app.source_scroll.min(max_scroll);
    let source_visible: Vec<String> = wrapped_lines
        .iter()
        .skip(scroll)
        .take(visible_height)
        .map(line_to_string)
        .collect();

    let min_list_height = 6u16;
    let log_height = 6u16;
    let max_square = body[1]
        .height
        .saturating_sub(min_list_height + log_height);
    let preferred_height = body[1].width.saturating_div(2).max(3);
    let square_height = preferred_height.min(max_square.max(3));
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(square_height),
            Constraint::Min(min_list_height),
            Constraint::Length(log_height),
        ])
        .split(body[1]);

    app.list_page_size = right[1].height.saturating_sub(2).max(1) as usize;

    let vector_lines = build_vector_view_lines(app, right[0].width, right[0].height);
    let vector_visible: Vec<String> =
        vector_lines.iter().map(line_to_string).collect();

    let list_visible_height = right[1].height.saturating_sub(2).max(1) as usize;
    let list_visible = build_list_snapshot_lines(app, &list_indices, list_visible_height);

    let log_visible_height = right[2].height.saturating_sub(2).max(1) as usize;
    let mut log_lines: Vec<String> = if app.logs.is_empty() {
        vec!["log idle".to_string()]
    } else {
        app.logs.iter().cloned().collect()
    };
    if log_lines.len() > log_visible_height {
        log_lines = log_lines.split_off(log_lines.len() - log_visible_height);
    }

    let mut out = String::new();
    let mut display_status = app.status.clone();
    if let Some(label) = app.selected_terms_label() {
        display_status = format!("{} | {}", display_status, label);
    }
    let header_line = format!(
        "🚀 Vector's Might 💥 | {} | [{}]",
        display_status,
        build_search_display(app)
    );
    let ticker_line = build_term_ticker_line(app, layout[1].width);
    let ticker_text = line_to_string(&ticker_line);
    out.push_str(&header_line);
    out.push('\n');
    out.push_str(&ticker_text);
    out.push('\n');
    out.push_str(&format!("[{}]\n", source_title));
    for line in source_visible {
        out.push_str(&line);
        out.push('\n');
    }
    out.push('\n');
    out.push_str("[Multidimensional Scanner]\n");
    for line in vector_visible {
        out.push_str(&line);
        out.push('\n');
    }
    out.push('\n');
    out.push_str(&format!("[{}]\n", list_title));
    for line in list_visible {
        out.push_str(&line);
        out.push('\n');
    }
    out.push('\n');
    out.push_str("[Embed log]\n");
    for line in log_lines {
        out.push_str(&line);
        out.push('\n');
    }

    Ok(out)
}

fn current_list_indices(app: &App) -> Vec<usize> {
    let mut indices = if app.similarity_mode == SimilarityMode::Query
        && !app.search_result_indices.is_empty()
    {
        app.search_result_indices.clone()
    } else if app.similarity_mode == SimilarityMode::Browse
        && !app.browse_result_indices.is_empty()
    {
        app.browse_result_indices.clone()
    } else {
        (0..app.chunks.len()).collect()
    };

    if app.term_bias_active() {
        let has_term_scores = indices.iter().any(|idx| {
            app.term_scores
                .get(*idx)
                .copied()
                .unwrap_or(0.0)
                > 0.0
        });
        if has_term_scores {
            indices.sort_by(|a, b| {
                let term_a = app.term_scores.get(*a).copied().unwrap_or(0.0);
                let term_b = app.term_scores.get(*b).copied().unwrap_or(0.0);
                let term_cmp = term_b
                    .partial_cmp(&term_a)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if term_cmp != std::cmp::Ordering::Equal {
                    return term_cmp;
                }
                if term_a <= 0.0 && term_b <= 0.0 {
                    return a.cmp(b);
                }
                let sim_a = app.similarity_scores.get(*a).copied().unwrap_or(0.0);
                let sim_b = app.similarity_scores.get(*b).copied().unwrap_or(0.0);
                sim_b
                    .partial_cmp(&sim_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
        }
    }

    indices
}

struct EmbedderProcess {
    stdin: ChildStdin,
    rx: Receiver<EmbedEvent>,
    _child: Child,
}

struct EventHub {
    listeners: Mutex<Vec<mpsc::Sender<String>>>,
}

impl EventHub {
    fn new() -> Self {
        Self {
            listeners: Mutex::new(Vec::new()),
        }
    }

    fn add_listener(&self) -> Receiver<String> {
        let (tx, rx) = mpsc::channel();
        if let Ok(mut listeners) = self.listeners.lock() {
            listeners.push(tx);
        }
        rx
    }

    fn send(&self, message: &str) {
        let mut stale = Vec::new();
        if let Ok(mut listeners) = self.listeners.lock() {
            for (idx, sender) in listeners.iter().enumerate() {
                if sender.send(message.to_string()).is_err() {
                    stale.push(idx);
                }
            }
            for idx in stale.into_iter().rev() {
                listeners.swap_remove(idx);
            }
        }
    }
}

fn handle_event_stream(mut stream: TcpStream, hub: Arc<EventHub>) -> io::Result<()> {
    let mut reader = io::BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" {
            break;
        }
    }

    if path == "/" || path == "/index.html" {
        let body = EVENT_DASHBOARD_HTML;
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n",
            body.as_bytes().len()
        );
        stream.write_all(header.as_bytes())?;
        stream.write_all(body.as_bytes())?;
        stream.flush()?;
        return Ok(());
    }

    if path != "/events" && path != "/events/stream" {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        let _ = stream.flush();
        return Ok(());
    }

    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n",
    )?;
    stream.flush()?;

    let rx = hub.add_listener();
    for message in rx {
        let payload = format!("data: {}\n\n", message);
        if stream.write_all(payload.as_bytes()).is_err() {
            break;
        }
        if stream.flush().is_err() {
            break;
        }
    }

    Ok(())
}

fn start_event_server(hub: Arc<EventHub>, port: u16) -> io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let hub = hub.clone();
            thread::spawn(move || {
                let _ = handle_event_stream(stream, hub);
            });
        }
    });
    Ok(())
}

fn spawn_embedder_process(config: &ScoreConfig, keepalive: bool) -> Result<EmbedderProcess> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let embed_script = config.embed_script_path.clone();

    let (tx, rx) = mpsc::channel();
    let mut cmd = Command::new(python);
    cmd.arg(&embed_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if keepalive {
        cmd.env("VECTORS_MIGHT_KEEPALIVE", "1");
    }
    cmd.env("TOKENIZERS_PARALLELISM", "true");
    let mut child = cmd.spawn().context("failed to start embedder")?;

    let stdin = child.stdin.take().context("embedder stdin missing")?;
    if let Some(stdout) = child.stdout.take() {
        let tx_clone = tx.clone();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<EmbedEvent>(&line) {
                    let _ = tx_clone.send(event);
                }
            }
        });
    }

    Ok(EmbedderProcess {
        stdin,
        rx,
        _child: child,
    })
}

fn send_embed_request(
    process: &mut EmbedderProcess,
    texts: Vec<String>,
    config: &ScoreConfig,
) -> Result<()> {
    let request = EmbedRequest {
        model: config.model.clone(),
        max_length: config.max_length,
        batch_size: config.embed_batch_size,
        texts,
    };
    let payload = serde_json::to_string(&request)?;
    process.stdin.write_all(payload.as_bytes())?;
    process.stdin.write_all(b"\n")?;
    process.stdin.flush()?;
    Ok(())
}

fn spawn_model_server(config: &ScoreConfig) -> Result<ModelServerProcess> {
    let python = config
        .python_path
        .as_deref()
        .unwrap_or_else(|| Path::new("python"));
    let mut child = Command::new(python)
        .arg(&config.model_server_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn vec2text model server")?;

    let mut stdin = child
        .stdin
        .take()
        .context("failed to open model server stdin")?;
    let stdout = child
        .stdout
        .take()
        .context("failed to open model server stdout")?;

    let config_line = serde_json::to_string(&serde_json::json!({
        "model": config.model,
    }))?;
    writeln!(stdin, "{}", config_line)?;
    stdin.flush()?;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let reader = io::BufReader::new(stdout);
        for line in reader.lines().flatten() {
            if let Ok(event) = serde_json::from_str::<ModelServerEvent>(&line) {
                let _ = tx.send(event);
            }
        }
    });

    Ok(ModelServerProcess { child, stdin, rx })
}

fn send_invert_request(
    process: &mut ModelServerProcess,
    embedding: Vec<f32>,
    config: &ScoreConfig,
    request_id: Option<u64>,
) -> Result<()> {
    let payload = serde_json::json!({
        "command": "invert_embeddings",
        "embeddings": [embedding],
        "num_steps": INVERT_NUM_STEPS,
        "max_length": config.max_length,
        "id": request_id,
    });
    let line = serde_json::to_string(&payload)?;
    writeln!(process.stdin, "{}", line)?;
    process.stdin.flush()?;
    Ok(())
}

fn spawn_embedder(texts: Vec<String>, config: &ScoreConfig) -> Result<Receiver<EmbedEvent>> {
    let python = if let Some(path) = &config.python_path {
        path.clone()
    } else {
        PathBuf::from("python")
    };
    let embed_script = config.embed_script_path.clone();
    let request = EmbedRequest {
        model: config.model.clone(),
        max_length: config.max_length,
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
            if stdin.write_all(&payload).is_err() {
                let _ = tx.send(EmbedEvent::Error {
                    message: "failed to write embedder payload".to_string(),
                });
                return;
            }
        }

        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if let Ok(event) = serde_json::from_str::<EmbedEvent>(&line) {
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
                let _ = tx.send(EmbedEvent::Error {
                    message: format!("embedder exit {} {}", exit, stderr),
                });
            } else {
                let _ = tx.send(EmbedEvent::Done);
            }
        }
    });

    Ok(rx)
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
    let mut events_enabled = false;
    let mut events_port = DEFAULT_EVENTS_PORT;
    while let Some(arg) = args.next() {
        if arg == "--input" || arg == "-i" {
            if let Some(path) = args.next() {
                input_path = Some(PathBuf::from(path));
            }
        } else if arg == "--py" {
            if let Some(path) = args.next() {
                score_config.python_path = Some(PathBuf::from(path));
            }
        } else if arg == "--batch" || arg == "--embed-batch" {
            if let Some(value) = args.next() {
                if let Ok(batch) = value.parse::<usize>() {
                    score_config.embed_batch_size = batch.max(1);
                }
            }
        } else if arg == "--events" {
            events_enabled = true;
        } else if arg == "--events-port" {
            if let Some(value) = args.next() {
                if let Ok(port) = value.parse::<u16>() {
                    events_port = port;
                    events_enabled = true;
                }
            }
        }
    }

    let mut app = App::new(input_path, score_config)?;
    if events_enabled {
        let hub = Arc::new(EventHub::new());
        match start_event_server(hub.clone(), events_port) {
            Ok(()) => {
                app.event_hub = Some(hub);
                app.push_log(&format!("events streaming on 127.0.0.1:{}", events_port));
            }
            Err(err) => {
                app.push_log(&format!("events server failed: {}", err));
            }
        }
    }
    let mut embed_process: Option<EmbedderProcess> = None;
    let mut search_embedder: Option<EmbedderProcess> = None;
    let mut invert_server: Option<ModelServerProcess> = None;
    let mut terminal = setup_terminal()?;
    let tick_rate = Duration::from_millis(200);
    let mut last_tick = Instant::now();

    loop {
        if let Some(process) = &mut embed_process {
            let mut embed_done = false;
            let mut embed_error: Option<String> = None;
            while let Ok(event) = process.rx.try_recv() {
                match event {
                    EmbedEvent::Embedding { index, embedding } => {
                        app.mark_embed_ready();
                        if index < app.embeddings.len() {
                            app.pending_embeddings.push_back((index, embedding));
                        }
                    }
                    EmbedEvent::Log { message } => {
                        app.note_embed_event();
                        if embed_log_loading(&message) {
                            app.embed_loading = true;
                        }
                        if embed_log_ready(&message) {
                            app.mark_embed_ready();
                        }
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

            if embed_done {
                app.embed_done_pending = true;
            } else if let Some(message) = embed_error {
                app.processing = false;
                app.status = format!("embedder error: {}", message);
                app.push_log("embedder error");
                embed_process = None;
            }
        }

        if app.processing && embed_process.is_some() && app.embedder_should_restart() {
            app.embed_restarts = app.embed_restarts.saturating_add(1);
            app.embed_ready = false;
            app.embed_loading = false;
            app.embed_last_event = Some(Instant::now());
            app.embed_started_at = Some(Instant::now());
            app.pending_embeddings.clear();
            app.status = format!(
                "embedder load stalled; restarting {}/{}",
                app.embed_restarts,
                MAX_EMBED_RESTARTS
            );
            app.push_log("embedder load stalled; restarting");
            if let Some(process) = embed_process.as_mut() {
                let _ = process._child.kill();
            }
            embed_process = None;
            let texts: Vec<String> = app
                .chunks
                .iter()
                .map(|c| sanitize_for_embedding(&c.text))
                .collect();
            match spawn_embedder_process(&app.score_config, true) {
                Ok(mut process) => {
                    if let Err(err) = send_embed_request(&mut process, texts, &app.score_config) {
                        app.processing = false;
                        app.status = format!("embedder restart send failed: {}", err);
                        app.push_log("embedder restart send failed");
                    } else {
                        embed_process = Some(process);
                    }
                }
                Err(err) => {
                    app.processing = false;
                    app.status = format!("embedder restart failed: {}", err);
                    app.push_log("embedder restart failed");
                }
            }
        }

        if let Some(process) = &mut search_embedder {
            let mut search_done = false;
            let mut search_error: Option<String> = None;
            let mut search_embedding: Option<Vec<f32>> = None;
            while let Ok(event) = process.rx.try_recv() {
                match event {
                    EmbedEvent::Embedding { embedding, .. } => {
                        search_embedding = Some(embedding);
                    }
                    EmbedEvent::Log { message } => {
                        app.push_log(&format!("search: {}", message));
                    }
                    EmbedEvent::Done => {
                        search_done = true;
                    }
                    EmbedEvent::Error { message } => {
                        search_error = Some(message);
                    }
                }
            }

            if let Some(embedding) = search_embedding {
                app.apply_search_embedding(embedding);
            }

            if search_done {
                app.search_embed_inflight = false;
            } else if let Some(message) = search_error {
                app.search_embed_inflight = false;
                app.status = format!("search embedder error: {}", message);
                app.push_log("search embedder error");
            }
        }

        if app.processing && !app.pending_embeddings.is_empty() {
            if app.last_embed_tick.elapsed() >= Duration::from_millis(80) {
                if let Some((index, embedding)) = app.pending_embeddings.pop_front() {
                    if index < app.embeddings.len() {
                        if app.embeddings[index].is_none() {
                            app.processed_count += 1;
                        }
                        app.update_vector_point(index, &embedding);
                        let query_sim = if app.similarity_mode == SimilarityMode::Query {
                            app.search_embedding
                                .as_ref()
                                .map(|query| cosine_similarity(query, &embedding).max(0.0))
                        } else {
                            None
                        };
                        app.apply_inversion_cache(index, &embedding);
                        if let Some(chunk) = app.chunks.get(index) {
                            let cache_key = chunk_cache_key(&chunk.text);
                            app.embedding_cache.insert(cache_key, embedding.clone());
                            app.embedding_cache_dirty = true;
                        }
                        app.embeddings[index] = Some(embedding);
                        if app.invert_all_active {
                            app.queue_inversion(index);
                        }
                        app.update_auto_related(index);
                        if let Some(sim) = query_sim {
                            if let Some(score) = app.similarity_scores.get_mut(index) {
                                *score = sim;
                            }
                            app.rebuild_search_results();
                        }
                        app.active_index = Some(index);
                        app.status = format!(
                            "embedded {}/{} | active {}",
                            app.processed_count,
                            app.chunks.len(),
                            index + 1
                        );
                        if app.similarity_mode == SimilarityMode::Browse {
                            app.compute_similarity_scores();
                        }
                        if index % 5 == 0 {
                            app.push_log(&format!("embedded chunk {}/{}", index + 1, app.chunks.len()));
                        }
                    }
                }
                app.last_embed_tick = Instant::now();
            }
        }

        if app.embed_done_pending && app.pending_embeddings.is_empty() && app.processing {
            app.processing = false;
            app.active_index = None;
            app.embed_started_at = None;
            app.last_embed_log = None;
            app.embed_done_pending = false;
            app.status = format!("embedding done | {} chunks", app.chunks.len());
            if app.similarity_mode == SimilarityMode::None {
                app.compute_similarity_groups();
                app.status = format!("embedding done | {} chunks", app.chunks.len());
            }
            app.push_log("embedding done");
            if app.embedding_cache_dirty {
                match save_embedding_cache(app.window_tokens, &app.embedding_cache) {
                    Ok(()) => {
                        app.embedding_cache_dirty = false;
                        app.push_log("embedding cache saved");
                    }
                    Err(err) => {
                        app.push_log(&format!("embedding cache save failed: {}", err));
                    }
                }
            }
        }

        if !app.processing {
            if let Some(delta) = app.pending_window_delta.take() {
                if delta != 0 {
                    app.adjust_window(delta);
                }
            }
        }

        if app.auto_embed_pending && !app.processing && !app.chunks.is_empty() {
            if !app.score_config.is_embed_enabled() {
                app.auto_embed_pending = false;
                app.status = "vec2text embed script missing".to_string();
            } else {
                app.processing = true;
                app.auto_embed_pending = false;
                app.embed_started_at = Some(Instant::now());
                app.last_embed_log = Some(Instant::now());
                app.embed_ready = false;
                app.embed_loading = false;
                app.embed_last_event = Some(Instant::now());
                app.embed_restarts = 0;
                app.processed_count = app.embeddings.iter().filter(|emb| emb.is_some()).count();
                app.pending_embeddings.clear();
                app.embed_done_pending = false;
                app.last_embed_tick = Instant::now();
                let texts: Vec<String> = app
                    .chunks
                    .iter()
                    .map(|c| sanitize_for_embedding(&c.text))
                    .collect();
                if embed_process.is_none() {
                    match spawn_embedder_process(&app.score_config, true) {
                        Ok(process) => {
                            embed_process = Some(process);
                        }
                        Err(err) => {
                            app.processing = false;
                            app.status = format!("embedder start failed: {}", err);
                        }
                    }
                }
                if let Some(process) = &mut embed_process {
                    match send_embed_request(process, texts, &app.score_config) {
                        Ok(()) => {
                            app.status = format!(
                                "embedding {}/{}",
                                app.processed_count,
                                app.chunks.len()
                            );
                            app.push_log(&format!("embedding {} chunks", app.chunks.len()));
                        }
                        Err(err) => {
                            app.processing = false;
                            app.status = format!("embedder send failed: {}", err);
                        }
                    }
                }
            }
        }

        if let Some(pending_idx) = app.extract_pending.take() {
            if invert_server.is_none() {
                match spawn_model_server(&app.score_config) {
                    Ok(server) => {
                        invert_server = Some(server);
                        app.invert_ready = false;
                        app.invert_loading = false;
                        app.invert_last_event = Some(Instant::now());
                    }
                    Err(err) => {
                        app.extract_status = format!("invert server start failed: {}", err);
                    }
                }
            }
            if let Some(server) = &mut invert_server {
                let embedding = app
                    .embeddings
                    .get(pending_idx)
                    .and_then(|emb| emb.as_ref())
                    .cloned();
                if let Some(embedding) = embedding {
                    app.extract_inflight = true;
                    app.extract_status = "inverting...".to_string();
                    if let Err(err) = send_invert_request(
                        server,
                        embedding,
                        &app.score_config,
                        Some(pending_idx as u64),
                    ) {
                        app.extract_inflight = false;
                        app.extract_status = format!("invert request failed: {}", err);
                    } else {
                        app.invert_inflight = Some(pending_idx);
                    }
                } else {
                    app.extract_inflight = false;
                    app.extract_status = "no embedding for selected chunk".to_string();
                }
            } else {
                app.extract_inflight = false;
            }
        }

        if app.invert_inflight.is_none() {
            if let Some(next_idx) = app.invert_queue.pop_front() {
                if let Some(flag) = app.invert_queued.get_mut(next_idx) {
                    *flag = false;
                }
                if invert_server.is_none() {
                    match spawn_model_server(&app.score_config) {
                        Ok(server) => {
                            invert_server = Some(server);
                            app.invert_ready = false;
                            app.invert_loading = false;
                            app.invert_last_event = Some(Instant::now());
                        }
                        Err(err) => {
                            app.invert_all_active = false;
                            app.status = format!("invert server start failed: {}", err);
                        }
                    }
                }
                if let Some(server) = &mut invert_server {
                    if let Some(embedding) = app
                        .embeddings
                        .get(next_idx)
                        .and_then(|emb| emb.as_ref())
                        .cloned()
                    {
                        if let Err(err) = send_invert_request(
                            server,
                            embedding,
                            &app.score_config,
                            Some(next_idx as u64),
                        ) {
                            app.invert_all_active = false;
                            app.status = format!("invert request failed: {}", err);
                        } else {
                            app.invert_inflight = Some(next_idx);
                            app.status = format!(
                                "inverting {}/{}",
                                app.invert_completed, app.invert_total
                            );
                        }
                    }
                }
            }
        }

        let mut drop_invert_server = false;
        if let Some(server) = &mut invert_server {
            while let Ok(event) = server.rx.try_recv() {
                match event {
                    ModelServerEvent::Ready => {
                        app.mark_invert_ready();
                        app.push_log("vec2text server ready");
                    }
                    ModelServerEvent::Log { message } => {
                        app.note_invert_event();
                        if invert_log_loading(&message) {
                            app.invert_loading = true;
                        }
                        if invert_log_ready(&message) {
                            app.mark_invert_ready();
                        }
                        app.push_log(&format!("vec2text: {}", message));
                    }
                    ModelServerEvent::Result {
                        command,
                        inversions,
                        scores,
                        id,
                        ..
                    } => {
                        app.mark_invert_ready();
                        if command == "invert_embeddings" {
                            let idx = id.and_then(|value| usize::try_from(value).ok());
                            let mut inversion_text: Option<String> = None;
                            let mut inversion_score: Option<f32> = None;
                            if let Some(mut values) = inversions {
                                inversion_text = values.pop();
                            }
                            if let Some(mut values) = scores {
                                inversion_score = values.pop();
                            }
                            if let Some(idx) = idx {
                                if let Some(score) = inversion_score {
                                    if let Some(slot) = app.invert_scores.get_mut(idx) {
                                        *slot = Some(score);
                                    }
                                }
                                if let Some(text) = inversion_text.clone() {
                                    if let Some(slot) = app.invert_texts.get_mut(idx) {
                                        *slot = Some(text.clone());
                                    }
                                    app.update_inversion_matches(idx, &text);
                                    if let Some(embedding) = app
                                        .embeddings
                                        .get(idx)
                                        .and_then(|emb| emb.as_ref())
                                    {
                                        let score_value = inversion_score.unwrap_or(0.0);
                                        app.invert_cache.insert(
                                            hash_embedding(embedding),
                                            CachedInversion {
                                                text: text.clone(),
                                                score: score_value,
                                            },
                                        );
                                        if let Err(err) = save_inversion_cache(&app.invert_cache) {
                                            app.push_log(&format!("cache save failed: {}", err));
                                        }
                                    }
                                }
                                if let Some(done) = app.invert_done.get_mut(idx) {
                                    if !*done {
                                        *done = true;
                                        app.invert_completed += 1;
                                    }
                                }
                                if app.invert_inflight == Some(idx) {
                                    app.invert_inflight = None;
                                }
                                if app.extract_chunk_index == Some(idx) {
                                    if let Some(text) = inversion_text {
                                        app.extract_text = Some(text);
                                    }
                                    if let Some(score) = inversion_score {
                                        app.extract_score = Some(score);
                                    }
                                    app.extract_status = "inversion ready".to_string();
                                    app.extract_inflight = false;
                                }
                            }
                            if app.invert_all_active {
                                if app.invert_completed >= app.invert_total {
                                    app.invert_all_active = false;
                                    app.status = format!(
                                        "inversion done | {} chunks",
                                        app.invert_completed
                                    );
                                } else {
                                    app.status = format!(
                                        "inverting {}/{}",
                                        app.invert_completed, app.invert_total
                                    );
                                }
                            }
                        }
                    }
                    ModelServerEvent::Error { message, id } => {
                        app.note_invert_event();
                        if let Some(idx) = id.and_then(|value| usize::try_from(value).ok()) {
                            if app.invert_inflight == Some(idx) {
                                app.invert_inflight = None;
                            }
                            if app.extract_chunk_index == Some(idx) {
                                app.extract_inflight = false;
                                app.extract_status = format!("invert error: {}", message);
                            }
                        } else {
                            app.extract_inflight = false;
                            app.extract_status = format!("invert error: {}", message);
                        }
                        app.push_log(&format!("invert error: {}", message));
                    }
                    ModelServerEvent::Shutdown { .. } => {
                        app.extract_inflight = false;
                        app.extract_status = "invert server stopped".to_string();
                        drop_invert_server = true;
                        break;
                    }
                }
            }
        }
        if drop_invert_server {
            invert_server = None;
        }

        if app.invert_should_restart() && invert_server.is_some() {
            app.invert_restarts = app.invert_restarts.saturating_add(1);
            app.invert_ready = false;
            app.invert_loading = false;
            app.invert_last_event = Some(Instant::now());
            app.status = format!(
                "invert server stalled; restarting {}/{}",
                app.invert_restarts,
                MAX_INVERT_RESTARTS
            );
            app.push_log("invert server stalled; restarting");
            if let Some(server) = invert_server.as_mut() {
                let _ = server.child.kill();
            }
            invert_server = None;
            if let Some(idx) = app.invert_inflight.take() {
                app.queue_inversion(idx);
            }
            if app.extract_inflight {
                if let Some(idx) = app.extract_chunk_index {
                    app.extract_pending = Some(idx);
                }
                app.extract_inflight = false;
                app.extract_status = "invert server reset".to_string();
            }
        }

        terminal.draw(|f| draw_ui(f, &mut app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                let status_before = app.status.clone();
                let was_processing = app.processing;
                let focus_before = app.focus_area;
                let key_code = key.code;
                let mut should_quit = false;

                if app.show_extract_view {
                    match key.code {
                        KeyCode::Esc | KeyCode::Enter => {
                            app.show_extract_view = false;
                            app.extract_chunk_index = None;
                        }
                        KeyCode::Up => {
                            app.extract_scroll = app.extract_scroll.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            app.extract_scroll = app.extract_scroll.saturating_add(1);
                        }
                        KeyCode::PageUp => {
                            app.extract_scroll = app.extract_scroll.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            app.extract_scroll = app.extract_scroll.saturating_add(10);
                        }
                        KeyCode::Home => {
                            app.extract_scroll = 0;
                        }
                        KeyCode::End => {
                            app.extract_scroll = usize::MAX;
                        }
                        KeyCode::Char('q') => {
                            should_quit = true;
                        }
                        _ => {}
                    }
                } else if app.show_similarity_menu {
                    match key.code {
                        KeyCode::Esc => {
                            app.show_similarity_menu = false;
                        }
                        KeyCode::Up => {
                            if app.similarity_menu_index > 0 {
                                app.similarity_menu_index -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if app.similarity_menu_index < 3 {
                                app.similarity_menu_index += 1;
                            }
                        }
                        KeyCode::Enter => {
                            match app.similarity_menu_index {
                                0 => app.start_browse_similarity(),
                                1 => app.compute_similarity_groups(),
                                2 => {
                                    if let Err(err) = app.open_extract_view() {
                                        app.status = format!("invert failed: {}", err);
                                    } else {
                                        app.status = "inversion view".to_string();
                                    }
                                }
                                3 => {
                                    app.toggle_prune_mark();
                                }
                                _ => {}
                            }
                            app.show_similarity_menu = false;
                        }
                        KeyCode::Char('q') => {
                            should_quit = true;
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Esc => {
                            if app.focus_area == FocusArea::Search {
                                app.search_query.clear();
                                app.search_query_snapshot = None;
                                app.search_matches.clear();
                                app.search_embedding = None;
                                app.search_filter = None;
                                app.search_result_indices.clear();
                                app.search_embed_inflight = false;
                                if app.similarity_mode == SimilarityMode::Query {
                                    app.similarity_mode = SimilarityMode::None;
                                    app.similarity_scores.fill(0.0);
                                }
                                app.focus_area = FocusArea::ChunkList;
                                app.status = "search closed".to_string();
                            } else if app.similarity_mode == SimilarityMode::Query {
                                app.clear_similarity();
                                app.focus_area = FocusArea::ChunkList;
                                app.status = "filter cleared".to_string();
                            } else if app.focus_area == FocusArea::TermTicker {
                                app.focus_area = FocusArea::ChunkList;
                                app.status = "chunk list - navigate with arrows".to_string();
                            } else if app.focus_area == FocusArea::SourceText {
                                app.focus_area = FocusArea::ChunkList;
                                app.status = "chunk list - navigate with arrows".to_string();
                            } else if app.similarity_mode != SimilarityMode::None {
                                app.clear_similarity();
                            }
                            if app.term_bias_active() {
                                app.reset_term_selection();
                            }
                        }
                        KeyCode::Enter => {
                            if app.focus_area == FocusArea::Search {
                                app.update_search();
                                app.focus_area = FocusArea::ChunkList;
                                app.search_query_snapshot = None;
                                let query = app.search_query.trim().to_string();
                                if query.is_empty() {
                                    app.status = "search empty".to_string();
                                } else if !app.score_config.is_embed_enabled() {
                                    app.status = "vec2text embed script missing".to_string();
                                } else if app.search_embed_inflight {
                                    app.status = "search in progress".to_string();
                                } else {
                                    if search_embedder.is_none() {
                                        match spawn_embedder_process(&app.score_config, true) {
                                            Ok(process) => {
                                                search_embedder = Some(process);
                                            }
                                            Err(err) => {
                                                app.status = format!("search embedder start failed: {}", err);
                                            }
                                        }
                                    }
                                    if let Some(process) = &mut search_embedder {
                                        app.search_embed_inflight = true;
                                        app.search_embedding = None;
                                        app.search_result_indices.clear();
                                        app.search_filter = Some(query.clone());
                                        app.similarity_scores.fill(0.0);
                                        app.status = format!("vectorizing search \"{}\"", query);
                                        app.push_log(&format!("filtering with \"{}\"", query));
                                        let sanitized_query = sanitize_for_embedding(&query);
                                        if sanitized_query.trim().is_empty() {
                                            app.search_embed_inflight = false;
                                            app.status = "search empty after sanitizing".to_string();
                                        } else if let Err(err) = send_embed_request(
                                            process,
                                            vec![sanitized_query],
                                            &app.score_config,
                                        ) {
                                            app.search_embed_inflight = false;
                                            app.status = format!("search embedder send failed: {}", err);
                                        }
                                    }
                                }
                            } else if app.focus_area == FocusArea::TermTicker {
                                let is_all = app
                                    .term_ticker
                                    .get(app.term_ticker_index)
                                    .map(|term| term.term == "All terms")
                                    .unwrap_or(false);
                                if is_all {
                                    app.reset_term_selection();
                                    app.status = "terms cleared".to_string();
                                    app.focus_area = FocusArea::ChunkList;
                                } else {
                                    app.focus_area = FocusArea::ChunkList;
                                    if !app.chunks.is_empty() {
                                        app.show_similarity_menu = true;
                                        app.similarity_menu_index = 0;
                                        app.status = "similarity menu".to_string();
                                    }
                                }
                            } else if app.focus_area == FocusArea::ChunkList && !app.chunks.is_empty() {
                                app.show_similarity_menu = true;
                                app.similarity_menu_index = 0;
                                app.status = "similarity menu".to_string();
                            }
                        }
                        KeyCode::Up => {
                            app.focus_area = FocusArea::ChunkList;
                            app.select_prev();
                        }
                        KeyCode::Down => {
                            app.focus_area = FocusArea::ChunkList;
                            app.select_next();
                        }
                        KeyCode::Left => {
                            if app.focus_area != FocusArea::Search {
                                app.focus_area = FocusArea::TermTicker;
                                app.select_term_prev();
                            }
                        }
                        KeyCode::Right => {
                            if app.focus_area != FocusArea::Search {
                                app.focus_area = FocusArea::TermTicker;
                                app.select_term_next();
                            }
                        }
                        KeyCode::PageUp => {
                            let step = app.source_page_size.max(1);
                            if app.source_scroll > 0 {
                                app.focus_area = FocusArea::SourceText;
                                app.source_scroll = app.source_scroll.saturating_sub(step);
                            } else {
                                app.focus_area = FocusArea::ChunkList;
                                app.page_chunk_list(false);
                            }
                        }
                        KeyCode::PageDown => {
                            let step = app.source_page_size.max(1);
                            if app.source_scroll < app.source_scroll_max {
                                app.focus_area = FocusArea::SourceText;
                                app.source_scroll = app.source_scroll.saturating_add(step);
                            } else {
                                app.focus_area = FocusArea::ChunkList;
                                app.page_chunk_list(true);
                            }
                        }
                        KeyCode::Home => {
                            app.focus_area = FocusArea::SourceText;
                            app.source_scroll = 0;
                        }
                        KeyCode::End => {
                            app.focus_area = FocusArea::SourceText;
                            app.source_scroll = usize::MAX;
                        }
                        KeyCode::Backspace => {
                            if app.focus_area == FocusArea::Search && !app.search_query.is_empty() {
                                app.search_query.pop();
                                app.update_search();
                            }
                        }
                        KeyCode::Delete => {
                            if app.focus_area == FocusArea::Search && !app.search_query.is_empty() {
                                app.search_query.pop();
                                app.update_search();
                            }
                        }
                        KeyCode::Char('/') => {
                            if app.focus_area == FocusArea::Search {
                                if app.search_query.chars().count() < SEARCH_BOX_WIDTH {
                                    app.search_query.push('/');
                                    app.update_search();
                                }
                            } else {
                                app.search_query_snapshot = Some(app.search_query.clone());
                                if app.search_query.chars().count() > SEARCH_BOX_WIDTH {
                                    app.search_query = app.search_query.chars().take(SEARCH_BOX_WIDTH).collect();
                                }
                                app.focus_area = FocusArea::Search;
                                app.status = "search mode - Enter submit, Esc cancel".to_string();
                            }
                        }
                        KeyCode::Char('\\') => {
                            if app.focus_area == FocusArea::Search {
                                if app.search_query.chars().count() < SEARCH_BOX_WIDTH {
                                    app.search_query.push('\\');
                                    app.update_search();
                                }
                            } else {
                                app.search_query_snapshot = Some(app.search_query.clone());
                                if app.search_query.chars().count() > SEARCH_BOX_WIDTH {
                                    app.search_query = app.search_query.chars().take(SEARCH_BOX_WIDTH).collect();
                                }
                                app.focus_area = FocusArea::Search;
                                app.status = "search mode - Enter submit, Esc cancel".to_string();
                            }
                        }
                        KeyCode::Char(c) => {
                            if app.focus_area == FocusArea::Search {
                                if app.search_query.chars().count() < SEARCH_BOX_WIDTH {
                                    app.search_query.push(c);
                                    app.update_search();
                                }
                            } else {
                                match c {
                                    'q' => {
                                        should_quit = true;
                                    }
                                    'r' => {
                                        if app.processing {
                                            if let Some(process) = embed_process.as_mut() {
                                                let _ = process._child.kill();
                                            }
                                            embed_process = None;
                                            app.processing = false;
                                            app.pending_embeddings.clear();
                                            app.embed_done_pending = false;
                                        }
                                        let _ = app.reload();
                                    }
                                    '+' | '=' => {
                                        app.adjust_scanner_zoom(SCANNER_ZOOM_STEP);
                                    }
                                    '-' => {
                                        app.adjust_scanner_zoom(-SCANNER_ZOOM_STEP);
                                    }
                                    '[' => {
                                        if app.processing {
                                            app.queue_window_adjust(-4);
                                        } else {
                                            app.adjust_window(-4);
                                        }
                                    }
                                    ']' => {
                                        if app.processing {
                                            app.queue_window_adjust(4);
                                        } else {
                                            app.adjust_window(4);
                                        }
                                    }
                                    'c' => {
                                        match build_screen_snapshot(&mut app) {
                                            Ok(text) => {
                                                if text.is_empty() {
                                                    app.status = "no text to copy".to_string();
                                                } else {
                                                    match copy_to_clipboard(&text) {
                                                        Ok(()) => {
                                                            app.status = format!(
                                                                "copied screen {} chars",
                                                                text.chars().count()
                                                            );
                                                        }
                                                        Err(err) => {
                                                            app.status = format!("copy failed: {}", err);
                                                        }
                                                    }
                                                }
                                            }
                                            Err(err) => {
                                                app.status = format!("copy failed: {}", err);
                                            }
                                        }
                                    }
                                    'i' => {
                                        match app.start_invert_all() {
                                            Ok(()) => {}
                                            Err(err) => {
                                                app.status = format!("invert all failed: {}", err);
                                            }
                                        }
                                    }
                                    'v' => {
                                        app.source_view = match app.source_view {
                                            SourceViewMode::Overview => SourceViewMode::Sorted,
                                            SourceViewMode::Sorted => SourceViewMode::Overview,
                                        };
                                        app.source_scroll = 0;
                                        app.status = match app.source_view {
                                            SourceViewMode::Overview => "source view".to_string(),
                                            SourceViewMode::Sorted => "sorted view".to_string(),
                                        };
                                    }
                                    'm' => {
                                        app.show_markers = !app.show_markers;
                                        app.status = if app.show_markers {
                                            "markers on".to_string()
                                        } else {
                                            "markers off".to_string()
                                        };
                                    }
                                    'p' => {
                                        app.toggle_prune_mark();
                                    }
                                    'P' => {
                                        app.apply_prune_marks();
                                    }
                                    't' => {
                                        app.time_enabled = !app.time_enabled;
                                        if app.time_enabled {
                                            app.orbit_speed = 0.03;
                                            app.status = "time on".to_string();
                                        } else {
                                            app.orbit_speed = 0.0;
                                            app.status = "time off".to_string();
                                        }
                                    }
                                    ' ' => {
                                        if app.focus_area == FocusArea::TermTicker {
                                            app.toggle_term_selection();
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if was_processing
                    && app.status != status_before
                    && !app.status.starts_with("processing")
                {
                    let skip_search_input = matches!(focus_before, FocusArea::Search)
                        && matches!(
                            key_code,
                            KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete
                        );
                    if !skip_search_input {
                        let status_message = app.status.clone();
                        app.push_log(&status_message);
                    }
                }

                if should_quit {
                    break;
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
            app.tick_animation();
            app.update_processing_status();
            app.update_gpu_status();
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

    restore_terminal(terminal)
}

fn build_term_ticker_line(app: &mut App, width: u16) -> Line<'static> {
    let available = width as usize;
    if available == 0 {
        return Line::from("");
    }
    if app.term_ticker.is_empty() {
        return Line::from("terms: (invert to populate)");
    }
    if app.term_ticker_index >= app.term_ticker.len() {
        app.term_ticker_index = 0;
    }

    #[derive(Clone)]
    struct Segment {
        text: String,
        style: Style,
        width: usize,
    }

    let label = "terms: ";
    let separator = " | ";
    let label_style = Style::default().fg(Color::DarkGray);
    let separator_style = Style::default().fg(Color::DarkGray);
    let normal_style = Style::default().fg(Color::Gray);
    let selected_style = Style::default()
        .fg(Color::Black)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    let mut segments: Vec<Segment> = Vec::new();
    let label_width = string_width(label);
    segments.push(Segment {
        text: label.to_string(),
        style: label_style,
        width: label_width,
    });

    let mut total_width = label_width;
    let mut selected_start = label_width;

    for (idx, term) in app.term_ticker.iter().enumerate() {
        let is_selected = app.selected_terms.iter().any(|item| item == &term.term);
        let base = if is_selected {
            format!("{}*", term.term)
        } else {
            term.term.clone()
        };
        let text = if term.term == "All terms" {
            base
        } else {
            format!("{}({})", base, term.count)
        };
        let width = string_width(&text);
        let style = if idx == app.term_ticker_index {
            selected_style
        } else {
            normal_style
        };
        let start = total_width;
        let end = start + width;
        if idx == app.term_ticker_index {
            selected_start = start;
        }
        segments.push(Segment { text, style, width });
        total_width = end;
        if idx + 1 < app.term_ticker.len() {
            let sep_width = string_width(separator);
            segments.push(Segment {
                text: separator.to_string(),
                style: separator_style,
                width: sep_width,
            });
            total_width += sep_width;
        }
    }

    let mut scroll = if total_width > available {
        selected_start
    } else {
        0
    };
    if total_width > available {
        scroll = scroll.min(total_width.saturating_sub(available));
    } else {
        scroll = 0;
    }
    app.term_ticker_scroll = scroll;

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut skip = scroll;
    let mut remaining = available;
    for segment in segments {
        if remaining == 0 {
            break;
        }
        if skip >= segment.width {
            skip -= segment.width;
            continue;
        }
        let slice = slice_by_width(&segment.text, skip, remaining);
        let used = string_width(&slice);
        if !slice.is_empty() {
            spans.push(Span::styled(slice, segment.style));
            remaining = remaining.saturating_sub(used);
        }
        skip = 0;
    }
    if remaining > 0 {
        spans.push(Span::raw(" ".repeat(remaining)));
    }

    Line::from(spans)
}

fn draw_ui(f: &mut ratatui::Frame, app: &mut App) {
    let size = f.size();

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(size);

    let search_active = app.focus_area == FocusArea::Search;
    let search_style = if search_active {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().fg(Color::Gray)
    };
    let search_spans = build_search_spans(app, search_style);
    let gpu_spans = build_gpu_spans(app);

    let status_style = Style::default().fg(Color::Rgb(255, 165, 0));
    let mut display_status = app.status.clone();
    if let Some(label) = app.selected_terms_label() {
        display_status = format!("{} | {}", display_status, label);
    }
    let mut title_spans = vec![
        Span::styled(
            " 🚀 Vector's Might 💥",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" | ", status_style),
        Span::styled(display_status, status_style),
        Span::styled(" | ", status_style),
    ];
    title_spans.extend(gpu_spans);
    title_spans.push(Span::styled(" | ", status_style));
    title_spans.push(Span::styled("[", Style::default().fg(Color::DarkGray)));
    title_spans.extend(search_spans);
    title_spans.push(Span::styled("]", Style::default().fg(Color::DarkGray)));
    let title = Line::from(title_spans);
    let header = Paragraph::new(Text::from(title))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, layout[0]);

    let ticker_line = build_term_ticker_line(app, layout[1].width);
    let ticker = Paragraph::new(Text::from(ticker_line));
    f.render_widget(ticker, layout[1]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[2]);

    let use_query_list = app.similarity_mode == SimilarityMode::Query
        && !app.search_result_indices.is_empty();
    let use_browse_list = app.similarity_mode == SimilarityMode::Browse
        && !app.browse_result_indices.is_empty();
    let list_indices = current_list_indices(app);
    let source_indices = if app.source_view == SourceViewMode::Sorted
        && (use_query_list || use_browse_list)
    {
        list_indices
            .iter()
            .cloned()
            .take(FILTERED_TEXT_LIMIT)
            .collect()
    } else {
        list_indices.clone()
    };
    let list_title = if app.similarity_mode == SimilarityMode::Query {
        if let Some(filter) = &app.search_filter {
            format!("Chunks | filtering with \"{}\"", filter)
        } else {
            "Chunks".to_string()
        }
    } else if app.similarity_mode == SimilarityMode::Browse {
        let anchor = app.browse_anchor.unwrap_or(app.selected);
        format!("Chunks | similar to {:03}", anchor + 1)
    } else {
        "Chunks".to_string()
    };

    let active_indices: HashSet<usize> = if app.source_view == SourceViewMode::Sorted {
        source_indices.iter().copied().collect()
    } else {
        (0..app.chunks.len()).collect()
    };
    if app.time_enabled {
        let scanner_points = build_scanner_points(app);
        let shimmer_levels = build_shimmer_levels(app, &scanner_points);
        app.emit_blink_events(&shimmer_levels, &active_indices);
    }

    let inner_width = body[0].width.saturating_sub(2) as usize;
    let visible_height = body[0].height.saturating_sub(2).max(1) as usize;
    let (source_lines, selected_marker_index, source_title) = match app.source_view {
        SourceViewMode::Overview => {
            let mut chunks = app.chunks.clone();
            chunks.sort_by_key(|chunk| chunk.start);
            let marker_index = chunks.iter().position(|chunk| chunk.index == app.selected);
            (build_overview_lines(app), marker_index, "Source Text")
        }
        SourceViewMode::Sorted => {
            let marker_index = source_indices.iter().position(|idx| *idx == app.selected);
            let title = if use_query_list || use_browse_list {
                "Filtered Text"
            } else {
                "Sorted Text"
            };
            (build_sorted_lines(app, &source_indices), marker_index, title)
        }
    };

    let (wrapped_lines, selected_line) = wrap_lines_with_marker(
        source_lines,
        inner_width.max(1),
        selected_marker_index,
        app.show_markers,
    );
    let line_count = wrapped_lines.len();

    let max_scroll = line_count.saturating_sub(visible_height);
    app.source_page_size = visible_height.max(1);
    app.source_scroll_max = max_scroll;
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

    let overview = Paragraph::new(Text::from(wrapped_lines))
        .block(Block::default().borders(Borders::ALL).title(source_title))
        .scroll((scroll_offset, 0));
    f.render_widget(overview, body[0]);
    app.source_scroll = new_scroll;

    let min_list_height = 6u16;
    let log_height = 6u16;
    let max_square = body[1]
        .height
        .saturating_sub(min_list_height + log_height);
    let preferred_height = body[1].width.saturating_div(2).max(3);
    let square_height = preferred_height.min(max_square.max(3));

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(square_height),
            Constraint::Min(min_list_height),
            Constraint::Length(log_height),
        ])
        .split(body[1]);

    app.list_page_size = right[1].height.saturating_sub(2).max(1) as usize;

    let vector_lines = build_vector_view_lines(app, right[0].width, right[0].height);
    let vector_view = Paragraph::new(Text::from(vector_lines))
        .block(Block::default().borders(Borders::ALL).title("Multidimensional Scanner"));
    f.render_widget(vector_view, right[0]);

    let items: Vec<ListItem> = list_indices
        .iter()
        .filter_map(|idx| app.chunks.get(*idx))
        .map(|chunk| {
            let sim = app
                .similarity_scores
                .get(chunk.index)
                .copied()
                .unwrap_or(0.0);
            let sim_suffix = match app.similarity_mode {
                SimilarityMode::Browse | SimilarityMode::Query => {
                    format!(" sim:{:.2}", sim)
                }
                _ => String::new(),
            };
            let term_suffix = if app.term_bias_active() {
                let term_score = app.term_scores.get(chunk.index).copied().unwrap_or(0.0);
                if term_score > 0.0 {
                    format!(" term:{:.0}", term_score)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            let prune_suffix = if app.prune_available(chunk.index) {
                if app
                    .invert_prune_marked
                    .get(chunk.index)
                    .copied()
                    .unwrap_or(false)
                {
                    " !"
                } else {
                    " *"
                }
            } else {
                ""
            };
            let line = format!(
                "{:>3} [{:>5}-{:>5}] {}{}{}{}",
                chunk.index + 1,
                chunk.start,
                chunk.end,
                chunk.preview,
                sim_suffix,
                term_suffix,
                prune_suffix
            );
            ListItem::new(line).style(list_chunk_style(app, chunk, app.selected == chunk.index))
        })
        .collect();
    let chunks_title = Span::raw(list_title);
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(chunks_title))
        .highlight_symbol("▶ ");
    if use_query_list || use_browse_list {
        let mut temp_state = ListState::default();
        if !list_indices.is_empty() {
            let selected_pos = list_indices
                .iter()
                .position(|idx| *idx == app.selected)
                .unwrap_or(0);
            temp_state.select(Some(selected_pos));
        }
        f.render_stateful_widget(list, right[1], &mut temp_state);
    } else {
        f.render_stateful_widget(list, right[1], &mut app.list_state);
    }

    let log_lines: Vec<Line> = if app.logs.is_empty() {
        vec![Line::from("log idle")]
    } else {
        app.logs.iter().map(|line| Line::from(line.as_str())).collect()
    };
    let log = Paragraph::new(Text::from(log_lines))
        .block(Block::default().borders(Borders::ALL).title("Embed log"));
    f.render_widget(log, right[2]);

    if app.show_similarity_menu {
        let menu_items = [
            "Browse similar",
            "Rebuild similarity groups",
            "Invert text",
            "Mark prune",
        ];
        let max_width = body[1].width.saturating_sub(2).max(1);
        let max_height = body[1].height.saturating_sub(2).max(1);
        let menu_width = max_width;
        let menu_height = (menu_items.len() as u16 + 2).min(max_height);
        let menu_x = body[1].x + (body[1].width.saturating_sub(menu_width) / 2);
        let menu_y = body[1].y + (body[1].height.saturating_sub(menu_height) / 2);
        let area = Rect {
            x: menu_x,
            y: menu_y,
            width: menu_width,
            height: menu_height,
        };
        let items: Vec<ListItem> = menu_items
            .iter()
            .map(|item| ListItem::new(*item))
            .collect();
        let mut menu_state = ListState::default();
        menu_state.select(Some(app.similarity_menu_index));
        let menu = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Similarity"))
            .highlight_symbol("▶ ");
        f.render_widget(Clear, area);
        f.render_stateful_widget(menu, area, &mut menu_state);
    }

    if app.show_extract_view {
        if let Some(idx) = app.extract_chunk_index {
            if let Some(chunk) = app.chunks.get(idx) {
                let modal_width = (layout[2].width.saturating_mul(70) / 100).max(20);
                let modal_height = (layout[2].height.saturating_mul(60) / 100).max(6);
                let area = Rect {
                    x: layout[2].x + layout[2].width.saturating_sub(modal_width) / 2,
                    y: layout[2].y + layout[2].height.saturating_sub(modal_height) / 2,
                    width: modal_width,
                    height: modal_height,
                };
                if area.width >= 3 && area.height >= 3 {
                    let inner_width = area.width.saturating_sub(2) as usize;
                    let inner_height = area.height.saturating_sub(2) as usize;
                    let display_text = app
                        .extract_text
                        .as_deref()
                        .unwrap_or(&app.extract_status);
                    let lines = build_extract_lines(display_text);
                    let (wrapped_lines, _) =
                        wrap_lines_with_marker(lines, inner_width.max(1), None, true);
                    let max_scroll = wrapped_lines.len().saturating_sub(inner_height);
                    let scroll = app.extract_scroll.min(max_scroll);
                    app.extract_scroll = scroll;
                    let title = if let Some(score) = app.extract_score {
                        format!(
                            "Vec2text Inversion {:03} score {:.2} (Esc to close)",
                            chunk.index + 1,
                            score
                        )
                    } else if app.extract_inflight {
                        format!("Vec2text Inversion {:03} (inverting...)", chunk.index + 1)
                    } else {
                        format!("Vec2text Inversion {:03} (Esc to close)", chunk.index + 1)
                    };
                    let view = Paragraph::new(Text::from(wrapped_lines))
                        .block(Block::default().borders(Borders::ALL).title(title))
                        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
                    f.render_widget(Clear, area);
                    f.render_widget(view, area);
                }
            }
        }
    }

    let help_text = "q quit  r reload  / or \\ search  Enter submit/menu  Esc cancel search  PgUp/Dn scroll  Left/Right terms  +/- zoom  [ ] window  i invert  v view  c copy  m markers  p/P prune  t time";
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray));
    f.render_widget(help, layout[3]);
}
