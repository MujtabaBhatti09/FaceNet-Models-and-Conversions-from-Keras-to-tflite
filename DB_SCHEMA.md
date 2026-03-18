# AMS PostgreSQL Database Schema

> Compatible with **PostgreSQL 15+**. Uses `pgvector` extension for storing
> MobileFaceNet float32[128] embeddings as native vector columns — enabling
> future cosine-similarity search directly in the DB.

---

## 1. Extensions

```sql
-- pgvector: stores and queries float32 embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- pgcrypto: for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

---

## 2. Tables

### 2.1 `employees`

Stores registered employees and their face embeddings.

```sql
CREATE TABLE employees (
  id            UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT          NOT NULL,
  department    TEXT          NOT NULL,
  email         TEXT          UNIQUE,
  -- MobileFaceNet outputs 128-dim L2-normalised float32 vector
  embedding     vector(128)   NOT NULL,
  created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Index for ANN (approximate nearest-neighbour) cosine search
-- Useful if you move matching to the DB layer in the future
CREATE INDEX idx_employees_embedding
  ON employees
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 10);
```

> **Embedding note:** The mobile app stores embeddings as `number[]` (plain
> JSON array of 128 floats) in AsyncStorage. When syncing to the backend,
> send the array as a JSON array in the request body. The API casts it to
> `vector(128)` on insert.

---

### 2.2 `attendance_records`

Raw punches synced from mobile devices (one row per IN or OUT event).

```sql
CREATE TABLE attendance_records (
  id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id     UUID          NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
  employee_name   TEXT          NOT NULL,       -- denormalised for query speed
  marked_at       TIMESTAMPTZ   NOT NULL,
  type            TEXT          NOT NULL CHECK (type IN ('IN', 'OUT')),
  confidence      NUMERIC(5,4)  NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  lat             DOUBLE PRECISION,
  lng             DOUBLE PRECISION,
  synced          BOOLEAN       NOT NULL DEFAULT true,
  created_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Fast date-range + employee filters
CREATE INDEX idx_att_employee_date ON attendance_records (employee_id, marked_at);
CREATE INDEX idx_att_marked_at     ON attendance_records (marked_at DESC);
CREATE INDEX idx_att_type          ON attendance_records (type);
```

---

### 2.3 `auth_users` (admin panel users)

Separate from employees — for admin login only.

```sql
CREATE TABLE auth_users (
  id            UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT    UNIQUE NOT NULL,
  password_hash TEXT    NOT NULL,   -- bcrypt hash, never store plaintext
  role          TEXT    NOT NULL DEFAULT 'admin' CHECK (role IN ('admin', 'viewer')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 3. INSERT Queries

### 3.1 Register a new employee (with embedding)

```sql
-- The embedding comes from the mobile app as a JSON float array, e.g.:
-- [0.0231, -0.1142, 0.0887, ...]  (128 values, L2-normalised)

INSERT INTO employees (name, department, email, embedding)
VALUES (
  'Ayesha Khan',
  'Engineering',
  'ayesha@company.com',
  -- Cast JSON array → pgvector using array literal syntax
  '[0.0231,-0.1142,0.0887,0.0412,-0.0654,0.1023,...]'::vector(128)
)
RETURNING id, name, department, created_at;
```

**From your Node.js/Next.js API (using `pg` or Drizzle):**

```typescript
// embedding: number[]  — the L2-normalised array from the mobile app
await db.query(
  `INSERT INTO employees (name, department, email, embedding)
   VALUES ($1, $2, $3, $4::vector)
   RETURNING id`,
  [name, department, email, JSON.stringify(embedding)]
);
```

---

### 3.2 Sync an attendance punch from mobile

```sql
INSERT INTO attendance_records (
  id, employee_id, employee_name, marked_at, type, confidence, lat, lng, synced
)
VALUES (
  'att_abc123',               -- mobile-generated ID (or let DB gen with gen_random_uuid())
  'emp-uuid-here',
  'Ayesha Khan',
  '2026-03-18T08:02:00Z',
  'IN',
  0.9241,
  24.8607,
  67.0011,
  true
)
ON CONFLICT (id) DO NOTHING;  -- idempotent: safe to call multiple times
```

**Batch insert (from `POST /api/attendance/batch`):**

```sql
INSERT INTO attendance_records
  (id, employee_id, employee_name, marked_at, type, confidence, lat, lng, synced)
VALUES
  ($1, $2, $3, $4, $5, $6, $7, $8, true),
  ($9, $10, $11, $12, $13, $14, $15, $16, true)
  -- repeat per record
ON CONFLICT (id) DO NOTHING;
```

---

### 3.3 Update an employee's embedding (re-registration)

```sql
UPDATE employees
SET
  embedding  = '[0.0312,-0.0987,...]'::vector(128),
  updated_at = now()
WHERE id = 'emp-uuid-here';
```

---

## 4. SELECT Queries

### 4.1 Attendance for a date range (paired IN + OUT per day per employee)

This is the **core admin panel query** — it pairs IN and OUT punches into
single rows, matching what the `/api/attendance` route returns.

```sql
SELECT
  e.id                                          AS employee_id,
  e.name                                        AS employee_name,
  e.department,
  DATE(a_in.marked_at AT TIME ZONE 'UTC')       AS date,

  -- Check In
  a_in.marked_at                                AS check_in_time,
  a_in.confidence                               AS check_in_confidence,
  a_in.lat                                      AS check_in_lat,
  a_in.lng                                      AS check_in_lng,

  -- Check Out
  a_out.marked_at                               AS check_out_time,
  a_out.confidence                              AS check_out_confidence,
  a_out.lat                                     AS check_out_lat,
  a_out.lng                                     AS check_out_lng

FROM employees e

-- Left-join the earliest IN punch for each employee per day
LEFT JOIN LATERAL (
  SELECT * FROM attendance_records ar
  WHERE ar.employee_id = e.id
    AND ar.type = 'IN'
    AND ar.marked_at >= :from_date::date
    AND ar.marked_at <  :to_date::date + INTERVAL '1 day'
  ORDER BY ar.marked_at ASC
  LIMIT 1
) a_in ON true

-- Left-join the latest OUT punch for each employee per day
LEFT JOIN LATERAL (
  SELECT * FROM attendance_records ar
  WHERE ar.employee_id = e.id
    AND ar.type = 'OUT'
    AND DATE(ar.marked_at AT TIME ZONE 'UTC') = DATE(a_in.marked_at AT TIME ZONE 'UTC')
  ORDER BY ar.marked_at DESC
  LIMIT 1
) a_out ON true

WHERE
  a_in.id IS NOT NULL   -- at least checked in

ORDER BY
  date DESC,
  a_in.marked_at ASC;
```

**Parameters:** `:from_date` and `:to_date` as `YYYY-MM-DD` strings.

---

### 4.2 Today's summary stats

```sql
SELECT
  COUNT(DISTINCT e.id)                                        AS total_employees,
  COUNT(DISTINCT a_in.employee_id)                            AS present_today,
  COUNT(DISTINCT a_in.employee_id)
    FILTER (WHERE a_out.employee_id IS NOT NULL)              AS checked_out_today,
  COUNT(DISTINCT a_in.employee_id)
    FILTER (WHERE a_out.employee_id IS NULL)                  AS still_in_office

FROM employees e

LEFT JOIN attendance_records a_in
  ON a_in.employee_id = e.id
  AND a_in.type = 'IN'
  AND DATE(a_in.marked_at AT TIME ZONE 'UTC') = CURRENT_DATE

LEFT JOIN attendance_records a_out
  ON a_out.employee_id = e.id
  AND a_out.type = 'OUT'
  AND DATE(a_out.marked_at AT TIME ZONE 'UTC') = CURRENT_DATE;
```

---

### 4.3 All employees with embedding (for mobile sync)

```sql
-- Returns employees + embedding as JSON float array
-- The mobile app's syncEmployeesFromBackend() expects this shape
SELECT
  id,
  name,
  department,
  updated_at,
  embedding::float4[]   AS embedding   -- cast vector → float4[] for JSON serialisation
FROM employees
ORDER BY name;
```

> In Node.js, `pgvector` returns the vector as a JS `number[]` automatically
> if you use the `pgvector` npm package. Otherwise cast to `float4[]` as above.

---

### 4.4 Monthly report per employee

```sql
SELECT
  e.name,
  e.department,
  DATE(ar.marked_at AT TIME ZONE 'UTC')        AS date,
  MIN(ar.marked_at) FILTER (WHERE ar.type = 'IN')   AS check_in,
  MAX(ar.marked_at) FILTER (WHERE ar.type = 'OUT')  AS check_out,
  EXTRACT(EPOCH FROM (
    MAX(ar.marked_at) FILTER (WHERE ar.type = 'OUT') -
    MIN(ar.marked_at) FILTER (WHERE ar.type = 'IN')
  )) / 3600                                    AS hours_worked
FROM attendance_records ar
JOIN employees e ON e.id = ar.employee_id
WHERE
  DATE_TRUNC('month', ar.marked_at) = DATE_TRUNC('month', :target_month::date)
GROUP BY e.id, e.name, e.department, DATE(ar.marked_at AT TIME ZONE 'UTC')
ORDER BY e.name, date;
```

---

### 4.5 Cosine similarity face search (pgvector — optional future feature)

If you move matching to the database instead of the mobile device:

```sql
-- Find best matching employee for a given query embedding
SELECT
  id,
  name,
  department,
  1 - (embedding <=> '[0.0231,-0.1142,0.0887,...]'::vector(128)) AS cosine_similarity
FROM employees
ORDER BY embedding <=> '[0.0231,-0.1142,0.0887,...]'::vector(128)
LIMIT 1;
```

> `<=>` is the pgvector cosine distance operator. `1 - distance = similarity`.
> Threshold: reject if `cosine_similarity < 0.82` (matches mobile app default).

---

## 5. Drizzle ORM Schema (optional)

If you're using Drizzle with `drizzle-orm/pg-core` and `pgvector/drizzle`:

```typescript
import { pgTable, uuid, text, doublePrecision, numeric, boolean, timestamp } from 'drizzle-orm/pg-core';
import { vector } from 'pgvector/drizzle';

export const employees = pgTable('employees', {
  id:         uuid('id').primaryKey().defaultRandom(),
  name:       text('name').notNull(),
  department: text('department').notNull(),
  email:      text('email').unique(),
  embedding:  vector('embedding', { dimensions: 128 }).notNull(),
  createdAt:  timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt:  timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
});

export const attendanceRecords = pgTable('attendance_records', {
  id:           uuid('id').primaryKey().defaultRandom(),
  employeeId:   uuid('employee_id').notNull().references(() => employees.id, { onDelete: 'cascade' }),
  employeeName: text('employee_name').notNull(),
  markedAt:     timestamp('marked_at', { withTimezone: true }).notNull(),
  type:         text('type', { enum: ['IN', 'OUT'] }).notNull(),
  confidence:   numeric('confidence', { precision: 5, scale: 4 }).notNull(),
  lat:          doublePrecision('lat'),
  lng:          doublePrecision('lng'),
  synced:       boolean('synced').notNull().default(true),
  createdAt:    timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});
```

---

## 6. Quick Setup

```bash
# 1. Create DB
createdb ams_db

# 2. Apply schema
psql ams_db < schema.sql

# 3. Install pgvector (Ubuntu/Debian)
sudo apt install postgresql-15-pgvector

# 4. Enable in your DB
psql ams_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Node.js pgvector support
npm install pgvector
```

---

*Embeddings are 128-dimensional L2-normalised float32 vectors produced by
MobileFaceNet running on-device via `react-native-fast-tflite`. The matching
threshold used in production is **cosine similarity ≥ 0.82**.*
