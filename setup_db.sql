-- ============================================
-- Setup Database Penjualan Getuk
-- Jalankan di DBeaver SQL Editor
-- ============================================

-- Insert user admin default
-- Password: admin123
INSERT INTO users (nama, username, password)
VALUES ('Administrator', 'admin', '$2b$12$KQC1lV3NVk8tgHWbEQA7EeTcYVcnRSFuJZs4ez56ha7uR6VJpM.8C')
ON CONFLICT (username) DO NOTHING;

-- Verifikasi tabel
SELECT 'users' AS tabel, COUNT(*) AS jumlah FROM users
UNION ALL
SELECT 'penjualan', COUNT(*) FROM penjualan;
