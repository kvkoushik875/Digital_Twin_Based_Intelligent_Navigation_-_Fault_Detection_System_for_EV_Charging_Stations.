CREATE TABLE health_assessment (
    health_id BIGSERIAL PRIMARY KEY,
    station_id BIGINT NOT NULL,
    station_status VARCHAR(50) NOT NULL,
    health_score DOUBLE PRECISION,
    total_faults INTEGER NOT NULL,
    critical_faults INTEGER NOT NULL,
    warning_faults INTEGER NOT NULL,
    assessment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_health_assessment_station_id ON health_assessment (station_id);
CREATE INDEX idx_health_assessment_status ON health_assessment (station_status);
