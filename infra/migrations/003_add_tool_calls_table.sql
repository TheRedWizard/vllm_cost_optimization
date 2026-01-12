-- ============================================================================
-- Add tool_calls table for auditing external API calls
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_calls (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    
    -- Request identification
    call_id CHAR(36) NOT NULL UNIQUE,  -- UUID
    
    -- Tool info
    tool_name VARCHAR(100) NOT NULL,       -- e.g., "wikipedia_search"
    tool_category VARCHAR(50) NOT NULL,    -- e.g., "wikipedia", "wikidata"
    
    -- Input (what was sent)
    input_params JSON NOT NULL,            -- All parameters passed to the tool
    
    -- Output (what was received)
    output_data JSON DEFAULT NULL,         -- The response data
    output_summary VARCHAR(500) DEFAULT NULL,  -- Human-readable summary
    
    -- Metrics
    response_time_ms INT UNSIGNED DEFAULT NULL,
    
    -- Status
    status ENUM('pending', 'success', 'error') DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    
    -- Link to chat request (if called as part of a chat)
    chat_request_id BIGINT UNSIGNED DEFAULT NULL,
    
    -- Timestamps
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) DEFAULT NULL,
    
    FOREIGN KEY (chat_request_id) REFERENCES chat_requests(id),
    
    INDEX idx_tool_name (tool_name),
    INDEX idx_tool_category (tool_category),
    INDEX idx_tool_status (status),
    INDEX idx_tool_created (created_at),
    INDEX idx_tool_chat_request (chat_request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
