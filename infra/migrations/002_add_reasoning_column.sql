-- ============================================================================
-- Add reasoning_content column to chat_requests
-- ============================================================================
-- Stores chain-of-thought reasoning separately from final content
-- for thinking models like DeepSeek-R1 and Qwen3

ALTER TABLE chat_requests
ADD COLUMN reasoning_content TEXT DEFAULT NULL AFTER response_content;

-- Add index for queries filtering by whether reasoning exists
ALTER TABLE chat_requests
ADD INDEX idx_chat_has_reasoning ((reasoning_content IS NOT NULL));
