-- ============================================================================
-- vLLM Cost Optimization - Initial Schema
-- ============================================================================
-- 
-- Design principles:
-- 1. IMMUTABILITY: Prompts are never modified, only versioned via content hash
-- 2. TRACEABILITY: Every request is logged with full context for reproducibility
-- 3. COMPOSABILITY: Flows can have branching/merging async steps
--

-- ----------------------------------------------------------------------------
-- PROMPTS: Immutable prompt storage with content-addressable hashing
-- ----------------------------------------------------------------------------
-- Once a prompt is created, it NEVER changes. If you need to modify a prompt,
-- create a new one. The hash ensures we can always reproduce exact results.

CREATE TABLE IF NOT EXISTS prompts (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- SHA-256 hash of content for deduplication and immutability
    content_hash CHAR(64) NOT NULL UNIQUE,
    
    -- The actual prompt text
    content TEXT NOT NULL,
    
    -- Optional human-readable name/label
    name VARCHAR(255) DEFAULT NULL,
    
    -- Optional description of what this prompt does
    description TEXT DEFAULT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_prompts_name (name),
    INDEX idx_prompts_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- MODELS: Registry of available models
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS models (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Model identifier (e.g., "qwen3:8b", "nomic-embed-text:latest")
    name VARCHAR(255) NOT NULL UNIQUE,
    
    -- Model type: 'chat' or 'embedding'
    model_type ENUM('chat', 'embedding') NOT NULL,
    
    -- Human-readable description
    description VARCHAR(500) DEFAULT NULL,
    
    -- Provider (e.g., "ollama", "openai")
    provider VARCHAR(100) DEFAULT 'ollama',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_models_type (model_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- CHAT_REQUESTS: Complete log of all chat/completion requests
-- ----------------------------------------------------------------------------
-- Everything needed to reproduce the exact same request

CREATE TABLE IF NOT EXISTS chat_requests (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Request identification
    request_id CHAR(36) NOT NULL UNIQUE,  -- UUID for external reference
    
    -- Model used
    model_id BIGINT UNSIGNED NOT NULL,
    model_name VARCHAR(255) NOT NULL,  -- Denormalized for easy querying
    
    -- System prompt (immutable reference)
    system_prompt_id BIGINT UNSIGNED DEFAULT NULL,
    
    -- User message (the actual input)
    user_message TEXT NOT NULL,
    
    -- Full messages array as JSON (for multi-turn conversations)
    messages_json JSON NOT NULL,
    
    -- Generation parameters
    temperature DECIMAL(3,2) DEFAULT 0.70,
    max_tokens INT UNSIGNED DEFAULT NULL,
    extra_params JSON DEFAULT NULL,  -- Any additional parameters
    
    -- Response
    response_content TEXT DEFAULT NULL,      -- Final answer/content
    reasoning_content TEXT DEFAULT NULL,     -- Chain-of-thought (thinking models)
    response_json JSON DEFAULT NULL,         -- Full API response for reproducibility
    
    -- Metrics
    response_time_ms INT UNSIGNED DEFAULT NULL,
    prompt_tokens INT UNSIGNED DEFAULT NULL,
    completion_tokens INT UNSIGNED DEFAULT NULL,
    total_tokens INT UNSIGNED DEFAULT NULL,
    
    -- Status
    status ENUM('pending', 'success', 'error') DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    
    -- Timestamps
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) DEFAULT NULL,
    
    -- Foreign keys
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (system_prompt_id) REFERENCES prompts(id),
    
    -- Indexes
    INDEX idx_chat_model (model_id),
    INDEX idx_chat_status (status),
    INDEX idx_chat_created (created_at),
    INDEX idx_chat_system_prompt (system_prompt_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- EMBEDDING_REQUESTS: Complete log of all embedding requests
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embedding_requests (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    request_id CHAR(36) NOT NULL UNIQUE,
    
    -- Model used
    model_id BIGINT UNSIGNED NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    
    -- Input (can be single text or array)
    input_text TEXT NOT NULL,           -- First/primary text
    input_json JSON NOT NULL,           -- Full input array
    input_count INT UNSIGNED NOT NULL,  -- Number of texts embedded
    
    -- Response
    dimensions INT UNSIGNED DEFAULT NULL,
    embeddings_json JSON DEFAULT NULL,  -- The actual embeddings (can be large!)
    
    -- Metrics
    response_time_ms INT UNSIGNED DEFAULT NULL,
    total_tokens INT UNSIGNED DEFAULT NULL,
    
    -- Status
    status ENUM('pending', 'success', 'error') DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    
    -- Timestamps
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) DEFAULT NULL,
    
    FOREIGN KEY (model_id) REFERENCES models(id),
    
    INDEX idx_embed_model (model_id),
    INDEX idx_embed_status (status),
    INDEX idx_embed_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- FLOWS: High-level task/workflow definitions
-- ----------------------------------------------------------------------------
-- A flow is a named sequence of prompt steps that can branch and merge

CREATE TABLE IF NOT EXISTS flows (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Unique identifier
    name VARCHAR(255) NOT NULL UNIQUE,
    
    -- Description of what this flow accomplishes
    description TEXT DEFAULT NULL,
    
    -- Version tracking (increment when flow structure changes)
    version INT UNSIGNED DEFAULT 1,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_flows_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- FLOW_STEPS: Individual steps within a flow
-- ----------------------------------------------------------------------------
-- Steps can have dependencies (parent steps that must complete first)

CREATE TABLE IF NOT EXISTS flow_steps (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    flow_id BIGINT UNSIGNED NOT NULL,
    
    -- Step identification
    step_key VARCHAR(100) NOT NULL,  -- Unique within flow (e.g., "analyze", "summarize")
    step_order INT UNSIGNED NOT NULL, -- Execution order hint
    
    -- What prompt to use (immutable reference)
    prompt_id BIGINT UNSIGNED NOT NULL,
    
    -- What model to use
    model_id BIGINT UNSIGNED NOT NULL,
    
    -- Step type
    step_type ENUM('chat', 'embedding') DEFAULT 'chat',
    
    -- Generation parameters for this step
    temperature DECIMAL(3,2) DEFAULT 0.70,
    max_tokens INT UNSIGNED DEFAULT NULL,
    
    -- Description
    description VARCHAR(500) DEFAULT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (flow_id) REFERENCES flows(id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id),
    FOREIGN KEY (model_id) REFERENCES models(id),
    
    UNIQUE KEY uk_flow_step (flow_id, step_key),
    INDEX idx_step_order (flow_id, step_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- FLOW_STEP_DEPENDENCIES: DAG of step dependencies
-- ----------------------------------------------------------------------------
-- Allows for branching and merging in flows

CREATE TABLE IF NOT EXISTS flow_step_dependencies (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- The step that has a dependency
    step_id BIGINT UNSIGNED NOT NULL,
    
    -- The step it depends on (must complete before step_id can run)
    depends_on_step_id BIGINT UNSIGNED NOT NULL,
    
    -- How to use the parent's output
    -- 'input': Parent output becomes input to this step
    -- 'context': Parent output is added to context
    dependency_type ENUM('input', 'context') DEFAULT 'input',
    
    FOREIGN KEY (step_id) REFERENCES flow_steps(id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_step_id) REFERENCES flow_steps(id) ON DELETE CASCADE,
    
    UNIQUE KEY uk_dependency (step_id, depends_on_step_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- FLOW_EXECUTIONS: Instances of flow runs
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS flow_executions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    execution_id CHAR(36) NOT NULL UNIQUE,  -- UUID
    
    flow_id BIGINT UNSIGNED NOT NULL,
    flow_version INT UNSIGNED NOT NULL,  -- Snapshot of flow version at execution time
    
    -- Initial input to the flow
    input_data JSON DEFAULT NULL,
    
    -- Final output (aggregated from terminal steps)
    output_data JSON DEFAULT NULL,
    
    -- Status
    status ENUM('pending', 'running', 'success', 'error', 'cancelled') DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    
    -- Timing
    started_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) DEFAULT NULL,
    
    FOREIGN KEY (flow_id) REFERENCES flows(id),
    
    INDEX idx_exec_flow (flow_id),
    INDEX idx_exec_status (status),
    INDEX idx_exec_started (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- FLOW_STEP_EXECUTIONS: Individual step execution within a flow run
-- ----------------------------------------------------------------------------
-- Links back to chat_requests/embedding_requests for full traceability

CREATE TABLE IF NOT EXISTS flow_step_executions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    flow_execution_id BIGINT UNSIGNED NOT NULL,
    flow_step_id BIGINT UNSIGNED NOT NULL,
    
    -- Link to the actual request (one of these will be set)
    chat_request_id BIGINT UNSIGNED DEFAULT NULL,
    embedding_request_id BIGINT UNSIGNED DEFAULT NULL,
    
    -- Input to this step (may be transformed from parent outputs)
    step_input JSON DEFAULT NULL,
    
    -- Output from this step
    step_output JSON DEFAULT NULL,
    
    -- Status
    status ENUM('pending', 'running', 'success', 'error', 'skipped') DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    
    -- Timing
    started_at TIMESTAMP(3) DEFAULT NULL,
    completed_at TIMESTAMP(3) DEFAULT NULL,
    
    FOREIGN KEY (flow_execution_id) REFERENCES flow_executions(id) ON DELETE CASCADE,
    FOREIGN KEY (flow_step_id) REFERENCES flow_steps(id),
    FOREIGN KEY (chat_request_id) REFERENCES chat_requests(id),
    FOREIGN KEY (embedding_request_id) REFERENCES embedding_requests(id),
    
    UNIQUE KEY uk_step_exec (flow_execution_id, flow_step_id),
    INDEX idx_step_exec_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ----------------------------------------------------------------------------
-- Insert default models from config
-- ----------------------------------------------------------------------------

INSERT INTO models (name, model_type, description, provider) VALUES
    ('qwen3:8b', 'chat', 'Think fast → general purpose, quick responses', 'ollama'),
    ('qwen2.5-coder:7b', 'chat', 'Write code → optimized for programming tasks', 'ollama'),
    ('mistral-nemo:latest', 'chat', 'Read a lot → 128k context window', 'ollama'),
    ('deepseek-r1:14b', 'chat', 'Think deeply → chain-of-thought reasoning', 'ollama'),
    ('qwen2.5vl:7b', 'chat', 'See images → vision-language model', 'ollama'),
    ('nomic-embed-text:latest', 'embedding', 'Search meaning → fast, 768 dims, 8k context', 'ollama'),
    ('bge-m3:latest', 'embedding', 'Search meaning → multilingual, 1024 dims', 'ollama')
ON DUPLICATE KEY UPDATE description = VALUES(description);
